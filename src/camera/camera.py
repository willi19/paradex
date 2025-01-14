import time
import json
from datetime import datetime
import numpy as np
from pathlib import Path
import PySpin as ps

import CameraConfig
import os

class Camera(CameraConfig):
    def __init__(
        self,
        camPtr,
        processor,
        lens_info,
        cam_info,
        root,
        port,
        saveVideo=False,
        syncMode=False,
    ):
        camPtr.Init()  # initialize camera
        self.device_nodemap = camPtr.GetTLDeviceNodeMap()  #
        self.nodeMap = camPtr.GetNodeMap()  #
        self.serialnum = self.get_serialnum()
        settingDict = lens_info[str(cam_info[self.serialnum]["lens"])]
        super().__init__(settingDict, saveVideo)

        self.cam = camPtr
        self.is_capturing = True
        self.is_recording = False

        self.record_flag = False
        self.captureCount = 0
        self.curframe = -1
        self.dropCount = 0
        self.fps = 0

        self.timestamps = dict([("timestamps", []), ("frameID", [])])
        self.is_time_corrected = False  # timestamp diff check
        self.prev_image = None

        self.frame_time_counter = np.zeros(10).astype(np.float64)

        self.syncMode = syncMode  # True : triggered, False : no trigger,
        self.saveVideo = saveVideo  # true : save in video, false : stream viewer

        # Check for spinview attribute for details

        self.configureSettings(self.nodeMap)

        self.image_processor = None
        # self.image_processor = ps.ImageProcessor()
        # self.image_processor.SetColorProcessing(ps.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

        # TODO : Mainly for video, timestamp
        self.outName = "PC" + str(port)  # number of c
        self.videoName = None
        self.root_path = root

        self.videoStream = ps.SpinVideo()
        video_option = ps.AVIOption()

        # # Set the video file format (e.g., MP4, AVI)
        video_option.frameRate = 30  # Set the desired frame rate
        video_option.height=1536
        video_option.width=2048
        self.videoOption = video_option
        self.cam.BeginAcquisition()  # Start acquiring images from camera

    def get_serialnum(self):
        serialnum_entry = self.device_nodemap.GetNode(
            "DeviceSerialNumber"
        )  # .GetValue()
        serialnum = ps.CStringPtr(serialnum_entry).GetValue()
        return serialnum

    def get_now(self):
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")

    def get_fps(self):
        if self.captureCount < 10:
            return 0
        else:
            duration = (
                self.frame_time_counter[(self.captureCount + 9) % 10]
                - self.frame_time_counter[self.captureCount % 10]
            )
            return 10 / duration

    def get_capture(self, return_img=True):
        retcode = False  # if pImageRaw is incomplete, return False
        if not self.is_time_corrected:

            #            before = time.time()

            pImageRaw = self.cam.GetNextImage()  # get from buffer
            framenum = pImageRaw.GetFrameID()
            
            if not pImageRaw.IsIncomplete():
                chunkData = pImageRaw.GetChunkData()
                # print("chunkd : ", time.time() - before)
                ts = chunkData.GetTimestamp()
                # print("tstamp : ", time.time() - before)
                if self.image_processor is not None:
                    pImageConv = self.image_processor.Convert(
                        pImageRaw, ps.PixelFormat_BayerRG8
                    )
                else:
                    pImageConv = pImageRaw
                # print("conveted : ", time.time() - before)
                if len(self.timestamps["timestamps"]) >= 2:
                    prior_ts = self.timestamps["timestamps"][-1]
                    # if ts - prior_ts > self.allowable_interval:
                    #     self.timestamps["timestamps"].append(
                    #         prior_ts + self.allowable_interval
                    #     )
                    #     self.timestamps["timestamps"].append(ts)
                    #     self.prev_image = pImageConv
                    #     retImage = np.zeros((1536, 2048, 3)).astype(np.uint8)
                    #     retcode = False
                    #     self.is_time_corrected = True
                    # else:
                    #     retImage = pImageConv
                    #     self.timestamps["timestamps"].append(ts)
                    #     retcode = True
                    retImage = pImageConv
                    self.timestamps["timestamps"].append(ts)
                    self.timestamps["frameID"].append(framenum)
                    retcode=True
                else:
                    retImage = pImageConv
                    self.timestamps["timestamps"].append(ts)
                    self.timestamps["frameID"].append(framenum)
                    retcode = True
                if self.is_recording:
                    try:
                        self.videoStream.Append(retImage)
                    except Exception as e:
                        print(e)
                self.frame_time_counter[self.captureCount % 10] = time.time()
                self.captureCount += 1
                self.curframe = framenum
                
            else:
                print(ps.Image_GetImageStatusDescription(pImageRaw.GetImageStatus()))
                retImage = None
                

            pImageRaw.Release()

            # else:
            #     # return black image
            #     retImage = np.zeros((1536,2048,3))
            #     retcode = False # do not save!
        else:
            retcode = True
            self.is_time_corrected = False
            retImage = self.prev_image

        # if self.captureCount % 10 == 0:
        #     self.fps = self.get_fps()
        # print("current fps : ", self.fps) for debugging
        return retImage, retcode

    def stop_camera(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam
        return

    # for saving file
    def set_record(self):
        if self.is_recording:
            print("Stop Recording")
            self.is_recording = False
            stampname = (
                self.outName + "_" + self.serialnum + self.get_now() + "_timestamp.json"
            )
            json.dump(
                self.timestamps, open(self.root_path + "/" + stampname, "w"), indent="\t"
            )
            self.videoStream.Close()
            print("Video Save finished")
        else:
            self.is_recording = True
            self.videoStream.SetMaximumFileSize(0)  # no limited size for the file
            self.videoName = self.outName + "_" + self.serialnum + self.get_now()
            savePath = self.root_path +"/" + self.videoName
            print("Start Recording")
            self.videoStream.Open(str(savePath), self.videoOption)
        return

    def configureSettings(self, nodeMap):
        self.configureGain(nodeMap)
        self.configureThroughPut(nodeMap)
        # configureTrigger(nodeMap)
        if not self.syncMode:
            self.configureFrameRate(nodeMap)  # we use trigger anyway
        else:
            self.configureTrigger(nodeMap)
        self.configureExposure(nodeMap)
        self.configureAcquisition(nodeMap)
        # Set Exposure time, Gain, Throughput limit, Trigger mode,
        self.configureChunk(nodeMap)  # getting timestamp
        return
