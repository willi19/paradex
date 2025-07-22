import time
import json
import PySpin as ps
import os
from threading import Event, Thread, Lock

class TimecodeReceiver():
    def __init__(
        self
    ):
        self.timestamps = dict([("timestamps", []), ("frameID", []), ("pc_time", [])])
        self.autoforce_ip()
        self.exit = Event()
        self.start_capture = Event()
        self.cam_start = Event()
        self.connect_flag = Event()
        self.save_end = Event()
        
        self.save_path = None
        self.lock = Lock()
        
        self.thread = Thread(target=self.run)
        self.thread.start()
        
        self.wait_for_connection()

    def autoforce_ip(self):
        system = ps.System.GetInstance()
        interfaceList = system.GetInterfaces() # virtual port included
        for pInterface in interfaceList:
            nodeMapInterface = pInterface.GetTLNodeMap()
            camera_list = pInterface.GetCameras()
            cam_num = len(camera_list)
            camera_list.Clear()

            if cam_num == 1:
                curIPNode = nodeMapInterface.GetNode("GevDeviceIPAddress")    
                if ps.IsAvailable(curIPNode) and ps.IsReadable(curIPNode):
                    ip_int = ps.CIntegerPtr(curIPNode).GetValue()
                
                ip_str = f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"
                if ip_str[:2] != "11":
                    ptrAutoForceIP = nodeMapInterface.GetNode("GevDeviceAutoForceIP")
                    if ps.IsAvailable(ptrAutoForceIP) and ps.IsWritable(ptrAutoForceIP) and ps.IsWritable(pInterface.TLInterface.DeviceSelector.GetAccessMode()):
                        pInterface.TLInterface.DeviceSelector.SetValue(0)
                        pInterface.TLInterface.GevDeviceAutoForceIP.Execute()

            del pInterface
            
        
        interfaceList.Clear()
        system.ReleaseInstance()
        return
    
    def wait_for_connection(self):
        self.connect_flag.wait()
        
    def wait_for_cam_start(self):
        self.cam_start.wait()
    
    def wait_for_save(self):
        self.save_end.wait()
        
    def start(self, save_path=None):
        self.save_path = save_path
            
        self.start_capture.set()
        self.wait_for_cam_start()
    
    def end(self):
        self.save_end.clear()
        self.start_capture.clear()
        self.wait_for_save()
        self.save_path = None
        
    def get_data(self):
        with self.lock:
            return self.cur_frame, self.frame_time
    
    def run(self):
        system = ps.System.GetInstance()
        cam_list = system.GetCameras()

        if cam_list.GetSize() < 1:
            print(f"No cameras found. Exiting...")
            cam_list.Clear()
            system.ReleaseInstance()
            return

        self.cam = cam_list.GetByIndex(0)
        self.cam .Init()  # initialize camera
        self.device_nodemap = self.cam.GetTLDeviceNodeMap()  #
        self.stream_nodemap = self.cam.GetTLStreamNodeMap()  #
        self.nodeMap = self.cam.GetNodeMap()  #
        
        self.configureSettings(self.nodeMap)
        self.connect_flag.set()
        
        self.cur_frame = -1
        self.frame_time = -1
        
        while not self.exit.is_set():
            while not self.start_capture.is_set():
                time.sleep(0.01)
                if self.exit.is_set():
                    break
            if self.exit.is_set():
                break
            
            self.cam.BeginAcquisition()
            self.cam_start.set()
            
            while self.start_capture.is_set():
                if self.exit.is_set():
                    break
                
                pImageRaw = self.cam.GetNextImage()  # get from buffer
                framenum = pImageRaw.GetFrameID()
                capture_time = time.time()
                
                if not pImageRaw.IsIncomplete():
                    chunkData = pImageRaw.GetChunkData()
                    ts = chunkData.GetTimestamp()
                    
                    self.timestamps["timestamps"].append(ts)
                    self.timestamps["frameID"].append(framenum)
                    self.timestamps["pc_time"].append(capture_time)
                    
                    with self.lock:
                        self.cur_frame = framenum
                        self.frame_time = capture_time
                        
                else:
                    print(ps.Image_GetImageStatusDescription(pImageRaw.GetImageStatus()))
            
                pImageRaw.Release()

            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
                json.dump(
                    self.timestamps, open(os.path.join(self.save_path,"camera_timestamp.json"), "w"), indent="\t"
                )
                
                self.timestamps = dict([("timestamps", []), ("frameID", []), ("pc_time", [])])
                
            self.cam.EndAcquisition()
            self.save_end.set()

        self.cam.DeInit()
        del self.cam
        
        cam_list.Clear()
        system.ReleaseInstance()

    def quit(self):
        """Stops the serial reader process."""
        if self.save_path is not None:
            self.end()
            
        self.exit.set()
        self.thread.join()

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
    
    def configureTrigger(self, nodeMap):
        print("trigger mode on")
        ptrTriggerMode = ps.CEnumerationPtr(nodeMap.GetNode("TriggerMode"))
        if not ps.IsReadable(ptrTriggerMode):
            # print("Unable to disable trigger mode (node retrieval). Aborting...")
            return False
        ptrTriggerModeOff = ps.CEnumEntryPtr(ptrTriggerMode.GetEntryByName("Off"))
        if not ps.IsReadable(ptrTriggerModeOff):
            # print("Unable to disable trigger mode (enum entry retrieval). Aborting...")
            return False
        ptrTriggerMode.SetIntValue(ptrTriggerModeOff.GetValue())
        ptrTriggerSelector = ps.CEnumerationPtr(nodeMap.GetNode("TriggerSelector"))
        if not ps.IsReadable(ptrTriggerSelector) or not ps.IsWritable(ptrTriggerSelector):
            # print("Unable to get or set trigger selector (node retrieval). Aborting...")
            return False
        ptrTriggerSelectorFrameStart = ps.CEnumEntryPtr(ptrTriggerSelector.GetEntryByName("FrameStart"))
        if not ps.IsReadable(ptrTriggerSelectorFrameStart):
            # print("Unable to get trigger selector FrameStart (enum entry retrieval). Aborting...")
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
    
    def configurePacketSize(self, nodeMap):
        ptrPayloadSize = ps.CIntegerPtr(nodeMap.GetNode("GevSCPSPacketSize"))
        if not ps.IsAvailable(ptrPayloadSize) or not ps.IsWritable(ptrPayloadSize):
            print("Unable to set packet size. Aborting...")
            return
        ptrPayloadSize.SetValue(9000)
        print("Packet size set to 9000")
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

        buffer_count.SetValue(10)
        print('Stream Buffer Count set to 10...')
        return 
    
    def configureSettings(self, nodeMap):
        self.configureThroughPut(nodeMap)
        self.configureTrigger(nodeMap)
        
        self.configureAcquisition(nodeMap)
        self.configureChunk(nodeMap)

        self.configureBuffer(self.stream_nodemap)
        self.configurePacketSize(self.nodeMap)