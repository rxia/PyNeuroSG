


for j in range(data_spk[date]['data'].shape[0]):  # loop for every trial (clip)
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    video_path = '{}/{}.m4v'.format(data_spk[date]['trial_info']['stim_paths'][j],
                                    data_spk[date]['trial_info']['stim_names'][j])
    video_path = video_path.replace('l:', '/shared/lab').replace('_bk', '_m4v')
    print((j,video_path))
    cap = cv2.VideoCapture(video_path)
    n_frame = np.floor(response_window[0]*frame_rate)
    while (n_frame>=np.floor(response_window[0]*frame_rate) and n_frame<=np.floor(response_window[1]*frame_rate)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is False:
            break