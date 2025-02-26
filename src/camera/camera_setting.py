import PySpin as ps


VIDEO_OPTION = {"uncompressed":ps.AVIOption,
                "mjpg":ps.MJPGOption,
                "h264":ps.H264Option}

TIME_LIMIT = {
    30.:4300000,

}

# Basic camera setting
class CameraConfig:
    def __init__(self, configDict, save_video=False):
        self.gain = configDict["Gain"]
        self.exp_time = configDict["Exposure"]
        self.frame_rate = configDict["fps"]
        self.allowable_interval = TIME_LIMIT[configDict["fps"]] #configDict["timelimit"]
        # TODO : Below are some default values, change later for further control
        self.height = 1536
        self.width = 2048 
        # Save into file via save option
        if save_video:
            self.videoStream = ps.SpinVideo()
            if "VideoFormat" not in configDict:
                self.videoOption = VIDEO_OPTION["uncompressed"]
            else:
                self.videoOption = VIDEO_OPTION[configDict["VideoFormat"]]
            self.videoOption.frameRate =  configDict["fps"]   
            self.videoOption.height = self.height
            self.videoOption.width = self.width
            # TODO : Below are some default values, change later for further control
            
            # if configDict["VideoFormat"] == "mjpg":
            #     self.videoOption.quality = 40
            # if configDict["VideoFormat"] == "h264":
            #     self.videoOption.bitrate = 1000000
        else:
            self.videoStream = None
            self.videoOption = None

        
    def configureGain(self, nodeMap):
        ptrValAuto = ps.CEnumerationPtr(nodeMap.GetNode("GainAuto"))
        if  not ps.IsReadable(ptrValAuto) or not ps.IsWritable(ptrValAuto):
            print("Unable to disable automatic gain (node retrieval). Aborting...")
            return            
        ptrValAutoOff = ps.CEnumEntryPtr(ptrValAuto.GetEntryByName("Off"))
        if not ps.IsReadable(ptrValAutoOff):
            print("Unable to disable automatic gain (enum entry retrieval). Aborting...")
            return
        ptrValAuto.SetIntValue(ptrValAutoOff.GetValue())
        ptrVal = ps.CFloatPtr(nodeMap.GetNode("Gain"))
        if not ps.IsReadable(ptrVal) or not ps.IsWritable(ptrVal):
            print("Unable to get or set gain. Aborting...")
            return
        # Ensure desired exposure time does not exceed the maximum
        ValMax = ptrVal.GetMax()
        if self.gain > ValMax: 
            self.gain = ValMax
        ptrVal.SetValue(self.gain)
        print("Gain set to ", self.gain)
        return

    def configureThroughPut(self, nodeMap):
        # Set throughput limit 
        ptrVal = ps.CIntegerPtr(nodeMap.GetNode("DeviceLinkThroughputLimit"))
        if not ps.IsReadable(ptrVal) or not ps.IsWritable(ptrVal):
            print("Unable to get or set throughput. Aborting...")
            return
        # Ensure desired exposure time does not exceed the maximum
        ValMax = ptrVal.GetMax()
        ValMin = ptrVal.GetMin()

        posValMax = ((ValMax - ValMin) // 16000) * 16000 + ValMin
        ptrVal.SetValue(posValMax)
        print("Throughput limit set to ", posValMax)
        return

    # if free-run mode
    def configureFrameRate(self, nodeMap):
        # If trigger mode is on, should turn this off
        ptrTriggerMode = ps.CEnumerationPtr(nodeMap.GetNode("TriggerMode"))
        if not ps.IsReadable(ptrTriggerMode):
            print("Unable to disable trigger mode (node retrieval). Aborting...")
            return
        ptrTriggerModeOff = ps.CEnumEntryPtr(ptrTriggerMode.GetEntryByName("Off"))
        if not ps.IsReadable(ptrTriggerModeOff):
            print("Unable to disable trigger mode (enum entry retrieval). Aborting...")
            return
        ptrTriggerMode.SetIntValue(ptrTriggerModeOff.GetValue())
        ptrValAuto = ps.CEnumerationPtr(nodeMap.GetNode("AcquisitionFrameRateAuto"))
        if not ps.IsReadable(ptrValAuto) or not ps.IsWritable(ptrValAuto):
            print("Unable to disable automatic framerate (node retrieval). Aborting...")
            return
        ptrValAutoOff = ps.CEnumEntryPtr(ptrValAuto.GetEntryByName("Off"))
        if not ps.IsReadable(ptrValAutoOff):
            print("Unable to disable automatic framerate (enum entry retrieval). Aborting...")
            return
        ptrValAuto.SetIntValue(ptrValAutoOff.GetValue())
        ptrVal = ps.CFloatPtr(nodeMap.GetNode("AcquisitionFrameRate"))
        if not ps.IsReadable(ptrVal) or not ps.IsWritable(ptrVal):
            print("Unable to get or set framerate. Aborting...")
            return
        
        # Ensure desired exposure time does not exceed the maximum
        # const double ValMax = ptrVal->GetMax();
        # if (gain > ValMax) gain = ValMax;
        ptrVal.SetValue(self.frame_rate)
        print("Framerate set to ",self.frame_rate)
        return

    # if Triggered mode
    def configureTrigger(self, nodeMap):
        ptrTriggerMode = ps.CEnumerationPtr(nodeMap.GetNode("TriggerMode"))
        if not ps.IsReadable(ptrTriggerMode):
            print("Unable to disable trigger mode (node retrieval). Aborting...")
            return
        ptrTriggerModeOff = ps.CEnumEntryPtr(ptrTriggerMode.GetEntryByName("Off"))
        if not ps.IsReadable(ptrTriggerModeOff):
            print("Unable to disable trigger mode (enum entry retrieval). Aborting...")
            return
        ptrTriggerMode.SetIntValue(ptrTriggerModeOff.GetValue())
        ptrTriggerSelector = ps.CEnumerationPtr(nodeMap.GetNode("TriggerSelector"))
        if not ps.IsReadable(ptrTriggerSelector) or not ps.IsWritable(ptrTriggerSelector):
            print("Unable to get or set trigger selector (node retrieval). Aborting...")
            return
        ptrTriggerSelectorFrameStart = ps.CEnumEntryPtr(ptrTriggerSelector.GetEntryByName("FrameStart"))
        if not ps.IsReadable(ptrTriggerSelectorFrameStart):
            print("Unable to get trigger selector FrameStart (enum entry retrieval). Aborting...")
            return
        
        # Set Frame start
        ptrTriggerSelector.SetIntValue(ptrTriggerSelectorFrameStart.GetValue())
        ptrTriggerSource = ps.CEnumerationPtr(nodeMap.GetNode("TriggerSource"))
        if not ps.IsReadable(ptrTriggerSource) or not ps.IsWritable(ptrTriggerSource):
            print("Unable to get or set trigger mode (node retrieval). Aborting...")
            return
        # Set Rising-edge trigger
        ptrTriggerActivation = ps.CEnumerationPtr(nodeMap.GetNode("TriggerActivation"))
        if not ps.IsReadable(ptrTriggerActivation) or not ps.IsWritable(ptrTriggerActivation):
            print("Unable to get or set trigger activation (node retrieval). Aborting...")
            return
        
        ptrRisingEdge = ps.CEnumEntryPtr(ptrTriggerActivation.GetEntryByName("RisingEdge"))
        if not ps.IsReadable(ptrRisingEdge):
            print("Unable to enable trigger mode RisingEdge (enum entry retrieval). Aborting...")
            return
        ptrTriggerActivation.SetIntValue(ptrRisingEdge.GetValue())

        # Set trigger mode to hardware ('Line0')
        ptrTriggerSourceHardware = ps.CEnumEntryPtr(ptrTriggerSource.GetEntryByName("Line0"))
        if not ps.IsReadable(ptrTriggerSourceHardware):
            print("Unable to set trigger mode Line0. Aborting...")
            return
        ptrTriggerSource.SetIntValue(ptrTriggerSourceHardware.GetValue())

        # Turn trigger mode on
        ptrTriggerModeOn = ps.CEnumEntryPtr(ptrTriggerMode.GetEntryByName("On"))
        if not ps.IsReadable(ptrTriggerModeOn):
            print("Unable to enable trigger On (enum entry retrieval). Aborting...")
            return
        ptrTriggerMode.SetIntValue(ptrTriggerModeOn.GetValue())

        # Set Trigger Overlap mode
        ptrTriggerOverlap = ps.CEnumerationPtr(nodeMap.GetNode("TriggerOverlap"))
        if not ps.IsReadable(ptrTriggerOverlap) or not ps.IsWritable(ptrTriggerOverlap):
            print("Unable to get or set trigger overlap (node retrieval). Aborting...")
            return        
        ptrReadOut = ps.CEnumEntryPtr(ptrTriggerOverlap.GetEntryByName("ReadOut"))
        if not ps.IsReadable(ptrReadOut):
            print("Unable to enable trigger Overlap readout(enum entry retrieval). Aborting...")
            return
        ptrTriggerOverlap.SetIntValue(ptrReadOut.GetValue())
        # NOTE: Blackfly and Flea3 GEV cameras need 1 second delay after trigger mode is turned on
        print("Trigger mode turned on...")
        return

    def configureExposure(self, nodeMap):
        ptrExposureAuto = ps.CEnumerationPtr(nodeMap.GetNode("ExposureAuto"))
        if not ps.IsReadable(ptrExposureAuto) or not ps.IsWritable(ptrExposureAuto):
            print("Unable to disable automatic exposure (node retrieval). Aborting...")
            return
        ptrExposureAutoOff = ps.CEnumEntryPtr(ptrExposureAuto.GetEntryByName("Off"))
        if not ps.IsReadable(ptrExposureAutoOff):
            print("Unable to disable automatic exposure (enum entry retrieval). Aborting...")
            return
        ptrExposureAuto.SetIntValue(ptrExposureAutoOff.GetValue())
        ptrExposureTime = ps.CFloatPtr(nodeMap.GetNode("ExposureTime"))
        if not ps.IsReadable(ptrExposureTime) or not ps.IsWritable(ptrExposureTime):
            print("Unable to get or set exposure time. Aborting...")
            return
        # Ensure desired exposure time does not exceed the maximum
        exposureTimeMax = ptrExposureTime.GetMax()
        if self.exp_time > exposureTimeMax: self.exp_time = exposureTimeMax
        ptrExposureTime.SetValue(self.exp_time)
        print("Exposure time set to ", self.exp_time,  " us...")
        return

    def configureAcquisition(self, nodeMap):
        ptrAcquisitionMode = ps.CEnumerationPtr(nodeMap.GetNode("AcquisitionMode"))
        if not ps.IsReadable(ptrAcquisitionMode) or not ps.IsWritable(ptrAcquisitionMode):
            print("Unable to set acquisition mode to continuous (enum retrieval). Aborting...")
            exit(-1)
        # Retrieve entry node from enumeration node
        ptrAcquisitionModeContinuous = ps.CEnumEntryPtr(ptrAcquisitionMode.GetEntryByName("Continuous"))
        if not ps.IsReadable(ptrAcquisitionModeContinuous): 
            print("Unable to get or set acquisition mode to continuous (entry retrieval). Aborting...")
            exit(-1)
        
        # Retrieve integer value from entry node
        acquisitionModeContinuous = ptrAcquisitionModeContinuous.GetValue()
        # Set integer value from entry node as new value of enumeration node
        ptrAcquisitionMode.SetIntValue(acquisitionModeContinuous)
        return

    def configureBuffer(self, nodeMap):
        handling_mode = ps.CEnumerationPtr(nodeMap.GetNode('StreamBufferHandlingMode'))
        if not ps.IsAvailable(handling_mode) or not ps.IsWritable(handling_mode):
            print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
            return False

        handling_mode_entry = ps.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not ps.IsAvailable(handling_mode_entry) or not ps.IsReadable(handling_mode_entry):
            print('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
            return False


        # Set stream buffer Count Mode to manual
        stream_buffer_count_mode = ps.CEnumerationPtr(nodeMap.GetNode('StreamBufferCountMode'))
        if not ps.IsAvailable(stream_buffer_count_mode) or not ps.IsWritable(stream_buffer_count_mode):
            print('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode_manual = ps.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Manual'))
        if not ps.IsAvailable(stream_buffer_count_mode_manual) or not ps.IsReadable(stream_buffer_count_mode_manual):
            print('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')
            return False

        stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
        print('Stream Buffer Count Mode set to manual...')

        # Retrieve and modify Stream Buffer Count
        buffer_count = ps.CIntegerPtr(nodeMap.GetNode('StreamBufferCountManual'))
        if not ps.IsAvailable(buffer_count) or not ps.IsWritable(buffer_count):
            print('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')
            return False

        buffer_count.SetValue(100)
        print('Stream Buffer Count set to 100...')
        return 

    def configurePacketSize(self, nodeMap):
        ptrPayloadSize = ps.CIntegerPtr(nodeMap.GetNode("GevSCPSPacketSize"))
        if not ps.IsAvailable(ptrPayloadSize) or not ps.IsWritable(ptrPayloadSize):
            print("Unable to set packet size. Aborting...")
            return
        ptrPayloadSize.SetValue(9000)
        print("Packet size set to 9000")
        return
    
    def configureChunk(self, nodeMap):
        ptrChunkModeActive = ps.CBooleanPtr(nodeMap.GetNode("ChunkModeActive"))
        if not ps.IsAvailable(ptrChunkModeActive) or not ps.IsWritable(ptrChunkModeActive):
            print("Camera is Unable to activate chunk mode. Aborting...")
            return
        
        ptrChunkModeActive.SetValue(True)
        ptrChunkSelector = ps.CEnumerationPtr(nodeMap.GetNode("ChunkSelector"))
        if not ps.IsAvailable(ptrChunkSelector) or not ps.IsWritable(ptrChunkSelector):
            print("Camera Chunk Selector is not writable")
            return

        # Select Timestamp for Chunk data
        ptrChunkSelectorTimestamp = ps.CEnumEntryPtr(ptrChunkSelector.GetEntryByName("Timestamp"))
        if not ps.IsAvailable(ptrChunkSelectorTimestamp) or not ps.IsReadable(ptrChunkSelectorTimestamp):
            print("Camera is Unable to set Chunk Selector (node retrieval). Aborting...")
            return
        
        ptrChunkSelector.SetIntValue(ptrChunkSelectorTimestamp.GetValue())
        # Else select also others
        ptrChunkEnable = ps.CBooleanPtr(nodeMap.GetNode("ChunkEnable"))
        # Retrieve corresponding boolean

        # Enable the boolean, thus enabling the corresponding chunk data
        if not ps.IsAvailable(ptrChunkEnable): print("Camera chunk not available")
        elif ptrChunkEnable.GetValue(): print("Camera chunk enabled")
        elif ps.IsWritable(ptrChunkEnable):
            ptrChunkEnable.SetValue(True)
            print("Camera chunk enabled")
        else:
            print("Camera chunk not writable")
        return
