import PySpin as ps

class Camera():
    def __init__(
        self,
        camPtr,
        gain,
        exposure_time,
        frame_rate=0.2,
        mode="video",
        syncMode=True,
    ):
        assert mode == "image" or mode == "video" or mode == "stream", "mode must be image or video"

        self.syncMode = syncMode  
        self.mode = mode
        self.gain = gain
        self.exposure_time = exposure_time
        self.frame_rate = frame_rate
        
        camPtr.Init()  # initialize camera
        self.cam = camPtr
        self.serialnum = self.get_serialnum()

        self.stream_nodemap = camPtr.GetTLStreamNodeMap() 
        self.nodeMap = camPtr.GetNodeMap() 
        
        self.configure()
        # return ret
    
    def configure(self):
        ret = True
        ret &= self.configureNodeSettings(self.nodeMap)
        ret &= self.configureBuffer(self.stream_nodemap)
        if not ret:
            raise RuntimeError("Failed to configure camera settings")
    
    def change_mode(self, mode, syncMode):
        self.mode = mode
        
        buffer_count = ps.CIntegerPtr(self.nodeMap.GetNode('StreamBufferCountManual'))
        if not ps.IsAvailable(buffer_count) or not ps.IsWritable(buffer_count):
            return False
        
        if self.mode == "video":
            buffer_count.SetValue(10)

        else:
            buffer_count.SetValue(1)
        self.configureAcquisition(self.nodeMap)
        
        if not syncMode and self.syncMode:
            ret &= self.configureFrameRate(self.nodeMap)  # we use trigger anyway
        elif not self.syncMode and syncMode:
            ret &= self.configureTrigger(self.nodeMap)
            
        self.syncMode = syncMode
        
        
        
    def get_serialnum(self):
        device_nodemap = self.cam.GetTLDeviceNodeMap()
        serialnum_entry = device_nodemap.GetNode(
            "DeviceSerialNumber"
        )  # .GetValue()
        serialnum = ps.CStringPtr(serialnum_entry).GetValue()
        return serialnum

    def get_image(self):
        while True:
            try:
                pImageRaw = self.cam.GetNextImage(100)
                return pImageRaw
            except:
                print(f"time out in {self.serialnum}")
                if self.mode == "image":
                    self.stop()
                    self.start()

        return pImageRaw

    def start(self):
        self.cam.BeginAcquisition()
        return
    
    def stop(self):
        self.cam.EndAcquisition()
        return
    
    def release(self):
        self.cam.DeInit()
        del self.cam
        return

    def configureNodeSettings(self, nodeMap):
        ret = True
        ret &= self.configureGain(nodeMap)
        ret &= self.configureThroughPut(nodeMap)
        # configureTrigger(nodeMap)
        if not self.syncMode:
            ret &= self.configureFrameRate(nodeMap)  # we use trigger anyway
        else:
            ret &= self.configureTrigger(nodeMap)
        
        # if self.mode == "video":
        #     ret &= self.configurePacketDelay(nodeMap)
        if not ret:
            print("Error configure trigger")

        ret &= self.configureExposure(nodeMap)
        if not ret:
            print("Error configure exposure")
        ret &= self.configureAcquisition(nodeMap)
        if not ret:
            print("Error configure Acquisition")
        # Set Exposure time, Gain, Throughput limit, Trigger mode,
        ret &= self.configureChunk(nodeMap)  # getting timestamp
        if not ret:
            print("Error configure chunk")
        # self.configureBuffer(nodeMap)
        ret &= self.configurePacketSize(self.nodeMap)
        if not ret:
            print("Error configure packet")
        return ret

    def configureGain(self, nodeMap):
        ptrValAuto = ps.CEnumerationPtr(nodeMap.GetNode("GainAuto"))
        if  not ps.IsReadable(ptrValAuto) or not ps.IsWritable(ptrValAuto):
            # print("Unable to disable automatic gain (node retrieval). Aborting...")
            return False            
        ptrValAutoOff = ps.CEnumEntryPtr(ptrValAuto.GetEntryByName("Off"))
        if not ps.IsReadable(ptrValAutoOff):
            # print("Unable to disable automatic gain (enum entry retrieval). Aborting...")
            return False
        
        ptrValAuto.SetIntValue(ptrValAutoOff.GetValue())
        ptrVal = ps.CFloatPtr(nodeMap.GetNode("Gain"))
        if not ps.IsReadable(ptrVal) or not ps.IsWritable(ptrVal):
            # print("Unable to get or set gain. Aborting...")
            return False
        # Ensure desired exposure time does not exceed the maximum
        ValMax = ptrVal.GetMax()
        if self.gain > ValMax: 
            self.gain = ValMax
        ptrVal.SetValue(self.gain)
        # print("Gain set to ", self.gain)
        return True

    def configureThroughPut(self, nodeMap):
        # Set throughput limit 
        ptrVal = ps.CIntegerPtr(nodeMap.GetNode("DeviceLinkThroughputLimit"))
        if not ps.IsReadable(ptrVal) or not ps.IsWritable(ptrVal):
            # print("Unable to get or set throughput. Aborting...")
            return False
        
        ValMax = ptrVal.GetMax()
        ValMin = ptrVal.GetMin()

        posValMax = ((ValMax - ValMin) // 16000) * 16000 + ValMin
        ptrVal.SetValue(posValMax)
        # print("Throughput limit set to ", posValMax)
        return True

    # if free-run mode
    def configureFrameRate(self, nodeMap):
        # If trigger mode is on, should turn this off
        ptrTriggerMode = ps.CEnumerationPtr(nodeMap.GetNode("TriggerMode"))
        if not ps.IsReadable(ptrTriggerMode):
            # print("Unable to disable trigger mode (node retrieval). Aborting...")
            return False
        
        ptrTriggerModeOff = ps.CEnumEntryPtr(ptrTriggerMode.GetEntryByName("Off"))
        if not ps.IsReadable(ptrTriggerModeOff):
            # print("Unable to disable trigger mode (enum entry retrieval). Aborting...")
            return False
        
        ptrTriggerMode.SetIntValue(ptrTriggerModeOff.GetValue())


        #############################################
        ptrValEnable = ps.CBooleanPtr(nodeMap.GetNode("AcquisitionFrameRateEnable"))
        if not ps.IsReadable(ptrValEnable) or not ps.IsWritable(ptrValEnable):
            # print("Unable to disable automatic framerate (node retrieval). Aborting...")
            return False
        ptrValEnable.SetValue(True)
        
        ptrVal = ps.CFloatPtr(nodeMap.GetNode("AcquisitionFrameRate"))
        if not ps.IsReadable(ptrVal) or not ps.IsWritable(ptrVal):
            # print("Unable to get or set framerate. Aborting...")
            return False
        
        # Ensure desired exposure time does not exceed the maximum
        # const double ValMax = ptrVal->GetMax();
        # if (gain > ValMax) gain = ValMax;
        ptrVal.SetValue(self.frame_rate)
        # print("Framerate set to ",self.frame_rate)
        return True

    # if Triggered mode
    def configureTrigger(self, nodeMap):
        ptrTriggerMode = ps.CEnumerationPtr(nodeMap.GetNode("TriggerMode"))
        if not ps.IsReadable(ptrTriggerMode):
            print("Unable to disable trigger mode (node retrieval). Aborting...")
            return False
        # ptrTriggerModeOff = ps.CEnumEntryPtr(ptrTriggerMode.GetEntryByName("Off"))
        # if not ps.IsReadable(ptrTriggerModeOff):
        #     print("Unable to disable trigger mode (enum entry retrieval). Aborting...")
        #     return False
        # ptrTriggerMode.SetIntValue(ptrTriggerModeOff.GetValue())
        ptrTriggerSelector = ps.CEnumerationPtr(nodeMap.GetNode("TriggerSelector"))
        if not ps.IsReadable(ptrTriggerSelector) or not ps.IsWritable(ptrTriggerSelector):
            print("Unable to get or set trigger selector (node retrieval). Aborting...")
            return False
        ptrTriggerSelectorFrameStart = ps.CEnumEntryPtr(ptrTriggerSelector.GetEntryByName("FrameStart"))
        if not ps.IsReadable(ptrTriggerSelectorFrameStart):
            print("Unable to get trigger selector FrameStart (enum entry retrieval). Aborting...")
            return False
        
        # Set Frame start
        ptrTriggerSelector.SetIntValue(ptrTriggerSelectorFrameStart.GetValue())
        ptrTriggerSource = ps.CEnumerationPtr(nodeMap.GetNode("TriggerSource"))
        if not ps.IsReadable(ptrTriggerSource) or not ps.IsWritable(ptrTriggerSource):
            print("Unable to get or set trigger mode (node retrieval). Aborting...")
            return False
        # Set Rising-edge trigger
        ptrTriggerActivation = ps.CEnumerationPtr(nodeMap.GetNode("TriggerActivation"))
        if not ps.IsReadable(ptrTriggerActivation) or not ps.IsWritable(ptrTriggerActivation):
            print("Unable to get or set trigger activation (node retrieval). Aborting...")
            return False
        
        ptrRisingEdge = ps.CEnumEntryPtr(ptrTriggerActivation.GetEntryByName("RisingEdge"))
        if not ps.IsReadable(ptrRisingEdge):
            print("Unable to enable trigger mode RisingEdge (enum entry retrieval). Aborting...")
            return False
        ptrTriggerActivation.SetIntValue(ptrRisingEdge.GetValue())

        # Set trigger mode to hardware ('Line0')
        ptrTriggerSourceHardware = ps.CEnumEntryPtr(ptrTriggerSource.GetEntryByName("Line0"))
        if not ps.IsReadable(ptrTriggerSourceHardware):
            print("Unable to set trigger mode Line0. Aborting...")
            return False
        ptrTriggerSource.SetIntValue(ptrTriggerSourceHardware.GetValue())

        # Turn trigger mode on
        ptrTriggerModeOn = ps.CEnumEntryPtr(ptrTriggerMode.GetEntryByName("On"))
        if not ps.IsReadable(ptrTriggerModeOn):
            print("Unable to enable trigger On (enum entry retrieval). Aborting...")
            return False
        ptrTriggerMode.SetIntValue(ptrTriggerModeOn.GetValue())

        # Set Trigger Overlap mode
        ptrTriggerOverlap = ps.CEnumerationPtr(nodeMap.GetNode("TriggerOverlap"))
        if not ps.IsReadable(ptrTriggerOverlap) or not ps.IsWritable(ptrTriggerOverlap):
            print("Unable to get or set trigger overlap (node retrieval). Aborting...")
            return False        
        ptrReadOut = ps.CEnumEntryPtr(ptrTriggerOverlap.GetEntryByName("ReadOut"))
        if not ps.IsReadable(ptrReadOut):
            print("Unable to enable trigger Overlap readout(enum entry retrieval). Aborting...")
            return False
        ptrTriggerOverlap.SetIntValue(ptrReadOut.GetValue())
        # NOTE: Blackfly and Flea3 GEV cameras need 1 second delay after trigger mode is turned on
        return True
    
    def configurePacketDelay(self, nodeMap):
        ptrPacketDelay = ps.CIntegerPtr(nodeMap.GetNode("GevSCPD"))
        if not ps.IsAvailable(ptrPacketDelay) or not ps.IsWritable(ptrPacketDelay):
            # print("Unable to set packet delay (node retrieval). Aborting...")
            return False
        ptrPacketDelay.SetValue(9000 * 2)
        return True
    
    def configureExposure(self, nodeMap):
        ptrExposureAuto = ps.CEnumerationPtr(nodeMap.GetNode("ExposureAuto"))
        if not ps.IsReadable(ptrExposureAuto) or not ps.IsWritable(ptrExposureAuto):
            print("Unable to disable automatic exposure (node retrieval). Aborting...")
            return False
        ptrExposureAutoOff = ps.CEnumEntryPtr(ptrExposureAuto.GetEntryByName("Off"))
        if not ps.IsReadable(ptrExposureAutoOff):
            print("Unable to disable automatic exposure (enum entry retrieval). Aborting...")
            return False
        ptrExposureAuto.SetIntValue(ptrExposureAutoOff.GetValue())
        ptrExposureTime = ps.CFloatPtr(nodeMap.GetNode("ExposureTime"))
        if not ps.IsReadable(ptrExposureTime) or not ps.IsWritable(ptrExposureTime):
            print("Unable to get or set exposure time. Aborting...")
            return False
        # Ensure desired exposure time does not exceed the maximum
        exposureTimeMax = ptrExposureTime.GetMax()
        if self.exposure_time > exposureTimeMax: self.exposure_time = exposureTimeMax
        ptrExposureTime.SetValue(self.exposure_time)
        # print("Exposure time set to ", self.exposure_time,  " us...")
        return True

    def configureAcquisition(self, nodeMap):
        ptrAcquisitionMode = ps.CEnumerationPtr(nodeMap.GetNode("AcquisitionMode"))
        if not ps.IsReadable(ptrAcquisitionMode) or not ps.IsWritable(ptrAcquisitionMode):
            #print("Unable to set acquisition mode to continuous (enum retrieval). Aborting...")
            return False
        # Retrieve entry node from enumeration node
        acquisitionmode = "SingleFrame" if self.mode == "image" else "Continuous"
        ptrAcquisitionModeVal = ps.CEnumEntryPtr(ptrAcquisitionMode.GetEntryByName(acquisitionmode))

        if not ps.IsReadable(ptrAcquisitionModeVal): 
            print("Unable to get or set acquisition mode to continuous (entry retrieval). Aborting...")
            return False
        
        # Retrieve integer value from entry node
        acquisitionModeContinuous = ptrAcquisitionModeVal.GetValue()
        # Set integer value from entry node as new value of enumeration node
        ptrAcquisitionMode.SetIntValue(acquisitionModeContinuous)
        return True

    def configureBuffer(self, nodeMap):
        handling_mode = ps.CEnumerationPtr(nodeMap.GetNode('StreamBufferHandlingMode'))
        if not ps.IsAvailable(handling_mode) or not ps.IsWritable(handling_mode):
            # print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            return False

        handling_mode_entry = ps.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not ps.IsAvailable(handling_mode_entry) or not ps.IsReadable(handling_mode_entry):
            # print('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
            return False


        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = ps.CEnumerationPtr(nodeMap.GetNode('StreamBufferCountMode'))
        if not ps.IsAvailable(stream_buffer_count_mode) or not ps.IsWritable(stream_buffer_count_mode):
            # print('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode_manual = ps.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
        if not ps.IsAvailable(stream_buffer_count_mode_manual) or not ps.IsReadable(stream_buffer_count_mode_manual):
            # print('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
        # print('Stream Buffer Count Mode set to manual...')

        # Retrieve and modify Stream Buffer Count
        buffer_count = ps.CIntegerPtr(nodeMap.GetNode('StreamBufferCountManual'))
        if not ps.IsAvailable(buffer_count) or not ps.IsWritable(buffer_count):
            # print('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')
            return False
        
        if self.mode == "video":
            buffer_count.SetValue(1)

        else:
            buffer_count.SetValue(1)
        return True

    def configurePacketSize(self, nodeMap):
        ptrPayloadSize = ps.CIntegerPtr(nodeMap.GetNode("GevSCPSPacketSize"))
        if not ps.IsAvailable(ptrPayloadSize) or not ps.IsWritable(ptrPayloadSize):
            return False

        ptrPayloadSize.SetValue(9000)        
        return True
    
    def configureChunk(self, nodeMap):
        ptrChunkModeActive = ps.CBooleanPtr(nodeMap.GetNode("ChunkModeActive"))
        if not ps.IsAvailable(ptrChunkModeActive) or not ps.IsWritable(ptrChunkModeActive):
            return False
        
        ptrChunkModeActive.SetValue(True)
        ptrChunkSelector = ps.CEnumerationPtr(nodeMap.GetNode("ChunkSelector"))
        if not ps.IsAvailable(ptrChunkSelector) or not ps.IsWritable(ptrChunkSelector):
            return False

        # Select Timestamp for Chunk data
        ptrChunkSelectorTimestamp = ps.CEnumEntryPtr(ptrChunkSelector.GetEntryByName("Timestamp"))
        if not ps.IsAvailable(ptrChunkSelectorTimestamp) or not ps.IsReadable(ptrChunkSelectorTimestamp):
            return False
        
        ptrChunkSelector.SetIntValue(ptrChunkSelectorTimestamp.GetValue())
        ptrChunkEnable = ps.CBooleanPtr(nodeMap.GetNode("ChunkEnable"))
        
        if not ps.IsAvailable(ptrChunkEnable) or not ps.IsWritable(ptrChunkEnable): 
            # print("Camera chunk not available")
            return False
        
        ptrChunkEnable.SetValue(True)
        # print("Camera chunk enabled")
        
        return True