import depthai
import cv2

with dai.Pipeline() as pipeline:
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(1280, 720)
    
    config = stereo.initialConfig

    # Median filter to remove the salt n pepper type pixels


    # Temporal Filter for flicker reduction
    # config.postProcessing.temporalFilter.enable = True
    # config.postProcessing.temporalFilter.alpha = 0.3
    # config.postProcessing.temporalFilter.delta = 5
    # config.postProcessing.temporalFilter.persistencyMode = dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4

    # Threshold Filter to remove invalid '0' pixels and set an operational range
    # config.postProcessing.thresholdFilter.minRange = 300  # 30cm
    config.postProcessing.thresholdFilter.maxRange = 6500 # 8.0m

    #config.setConfidenceThreshold(170)



    monoLeftOut = monoLeft.requestOutput((1280, 720))
    monoRightOut = monoRight.requestOutput((1280, 720))

    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    rightOut = monoRightOut.createOutputQueue()
    stereoOut = stereo.depth.createOutputQueue()

    pipeline.start()
    while pipeline.isRunning():
        stereoFrame = stereoOut.get()

        assert stereoFrame.validateTransformations()
        # depth = processDepthFrame(stereoFrame.getCvFrame())
        depth = stereoFrame.getCvFrame().astype(np.float32) / 1000.0
        cv2.imshow("aaa",depth)
        cv2.waitKey(1)

    pipeline.stop()