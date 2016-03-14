# File format specification

## Data files

All files are in HDF5.

  * The data are stored in the following files:
      
      * the **KWIK** file is the main file, it contains:
          * all metadata
          * spike times
          * clusters
          * recording for each spike time
          * probe-related information
          * information about channels
          * information about cluster groups
          * events, event_types
          * aesthetic information, user data, application data
      * the **KWX** file contains the **spiking data**: features, masks, waveforms
      * the **KWD** files contain the **raw/filtered recordings**
  
  * Once spike sorting is finished, one can discard the KWX and KWD files and just keep the KWIK file for subsequent analysis (where spike sorting information like features, waveforms... are not necessary).

  * All files contain a **version number** in `/` (`kwik_version` attribute), which is an integer equal to 2 now.

  * The input files the user provides to the programs to generate these data are:
  
      * the **raw data** coming out from the acquisition system, in any proprietary format (NS5, etc.)
      * processing parameters (PRM file) and description of the probe (PRB file)
  

### KWIK

Below is the structure of the KWIK file.Everything is a group, except fields with a star (*) which are either leaves (datasets: arrays or tables) or attributes of their parents.

[X] is 0, 1, 2...
    
    /kwik_version* [=2]
    /name*
    /application_data
        spikedetekt
            MY_SPIKEDETEKT_PARAM*
            ...
    /user_data
    /channel_groups
        [X]
            name*
            adjacency_graph* [Kx2 array of integers]
            application_data
            user_data
            channels
                [X]
                    name*
                    kwd_index*
                    ignored*
                    position* (in microns relative to the whole multishank probe)
                    voltage_gain* (in microvolts)
                    display_threshold*
                    application_data
                        klustaviewa
                        spikedetekt
                    user_data
            spikes
                time_samples* [N-long EArray of UInt64]
                time_fractional* [N-long EArray of UInt8]
                recording* [N-long EArray of UInt16]
                cluster* [N-long EArray of UInt32]
                cluster_original* [N-long EArray of UInt32]
                features_masks
                    hdf5_path* [='{kwx}/channel_groups/X/features_masks']
                waveforms_raw
                    hdf5_path* [='{kwx}/channel_groups/X/waveforms_raw']
                waveforms_filtered
                    hdf5_path* [='{kwx}/channel_groups/X/waveforms_filtered']
            clusters
                [X]
                    application_data
                        klustaviewa
                            color*
                    cluster_group*
                    mean_waveform_raw*
                    mean_waveform_filtered*
                    quality_measures
                        isolation_distance*
                        matrix_isolation*
                        refractory_violation*
                        amplitude*
                    user_data
                        ...
            cluster_groups
                [X]
                    name*
                    application_data
                        klustaviewa
                            color*
                    user_data
    /recordings
        [X]
            name*
            start_time*
            start_sample*
            sample_rate*
            bit_depth*
            band_high*
            band_low*
            raw
                hdf5_path* [='{raw.kwd}/recordings/X']
            high
                hdf5_path* [='{high.kwd}/recordings/X']
            low
                hdf5_path* [='{low.kwd}/recordings/X']
            user_data
    /event_types
        [X]
            user_data
            application_data
                klustaviewa
                    color*
            events
                time_samples* [N-long EArray of UInt64]
                recording* [N-long EArray of UInt16]
                user_data [group or EArray]

### KWX

The **KWX** file contains spike-sorting-related information.

    /channel_groups
        [X]
            features_masks* [(N x NFEATURES x 2) EArray of Float32]
            waveforms_raw* [(N x NWAVESAMPLES x NCHANNELS) EArray of Int16]
            waveforms_filtered* [(N x NWAVESAMPLES x NCHANNELS) EArray of Int16]

### KWD

The **KWD** files contain the original recordings (raw and filtered). Each file among the `.raw`, `.high` and `.low` contains:

    /recordings
        [X]
            data* [(NSAMPLES x NCHANNELS) EArray of Int16]
            filter
                name*
                param1*
            downsample_factor*


## User files

### PRB

This JSON text file describes the probe used for the experiment: its geometry, its topology, and the dead channels.

    {
        "channel_groups": 
            [
                {
                    "channels": [0, 1, 2, 3],
                    "graph": [[0, 1], [2, 3], ...],
                    "geometry": {"0": [0.1, 0.2], "1": [0.3, 0.4], ...}
                },
                {
                    "channels": [4, 5, 6, 7],
                    "graph": [[4, 5], [6, 7], ...],
                    "geometry": {"4": [0.1, 0.2], "5": [0.3, 0.4], ...}
                }
            ]
    }


### PRM

This Python script defines all parameters necessary for the programs to process, open and display the data.

    EXPERIMENT_NAME = 'myexperiment'
    RAW_DATA_FILES = ['n6mab041109blahblah1.ns5', 'n6mab041109blahblah2.ns5']
    PRB_FILE = 'buzsaki32.probe'
    NCHANNELS = 32
    SAMPLING_FREQUENCY = 20000.
    IGNORED_CHANNELS = [2, 5]
    NBITS = 16
    VOLTAGE_GAIN = 10.
    WAVEFORMS_NSAMPLES = 20  # or a dictionary {channel_group: nsamples}
    FETDIM = 3  # or a dictionary {channel_group: fetdim}
    # ...
    
    # SpikeDetekt parameters file
    # ...

