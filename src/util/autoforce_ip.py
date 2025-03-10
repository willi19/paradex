"""
AutoForce IP on cameras
"""
import PySpin as ps


def per_camera_forceip(camptr, interfptr):
    nodeMapInterface = interfptr.GetTLNodeMap()
    device_nodemap = camptr.GetTLDeviceNodeMap()
    serialnum_entry = device_nodemap.GetNode("DeviceSerialNumber")#.GetValue()
    serialnum = ps.CStringPtr(serialnum_entry).GetValue()
    print("Current Camera Serialnum : ", serialnum)
    ptrAutoForceIP = nodeMapInterface.GetNode("GevDeviceAutoForceIP")
    if ps.IsAvailable(ptrAutoForceIP) and ps.IsWritable(ptrAutoForceIP):
        if not ps.IsWritable(interfptr.TLInterface.DeviceSelector.GetAccessMode()):
            print("---Unable to write to the DeviceSelector node while forcing IP")
        else:
            interfptr.TLInterface.DeviceSelector.SetValue(0)
            interfptr.TLInterface.GevDeviceAutoForceIP.Execute()
            print("---AutoForceIP executed for camera ", serialnum)
    del ptrAutoForceIP
    return        


def main():
    system = ps.System.GetInstance()
    interfaceList = system.GetInterfaces() # virtual port included
    for pInterface in interfaceList:
        curCamList = pInterface.GetCameras()
        camSize = curCamList.GetSize()
        if camSize  == 1:
            for pCam in curCamList:
                per_camera_forceip(pCam, pInterface)
                del pCam
        elif camSize > 1:
            print("Something wrong in interface")
        else:
            continue
        curCamList.Clear()

    del pInterface
    interfaceList.Clear()
    system.ReleaseInstance()
    print("Safely released")
    return

if __name__ == "__main__":
    main()
