* spikedetekt2 can be used in two ways:

   * It offers a single command that accepts PRM/PRB files/dictionaries and performs the whole spike detection (in the future, it might also do clustering).

   * It offers a Python API to customize the whole process.

* Method 1 is for regular users, method 2 is for advanced users and especially for ourselves. It will make things simpler when we'll need to try different algorithms or workflows.

* The Python API in method 2 implements functions that simplify the way we read and write the data.

### Get a reader for the raw data

* Convention: by default, times are given in samples rather than seconds.

* The method `rd.to_seconds()` accepts a number, a tuple, a list, or an array with samples, 
  and returns the same in seconds.

      rd = RawDataReader('rawdata.ns5', nchannels=?)  # from a binary file
      rd = RawDataReader('experiment.kwik')  # from a kwik file
      rd = RawDataReader(rawdata)  # from a NumPy Nsamples x Nchannels array
      rd = RawDataReader(..., chunk_size=[in samples],
                            chunk_overlap=[in samples], 
                            sample_rate=[in Hz])
    
### Get a chunk of data

    chunk = rd.next_chunk()
    chunk.window_full == (s1, s2)  # with overlap
    chunk.window_keep == (s1, s2)  # without overlap
    chunk.data_chunk_full  # chunk_full_size x Nchannels array
    chunk.data_chunk_keep  # chunk_full_size x Nchannels array

    rd.reset()  # reset chunking and move the cursor to the beginning
    
### Get parameters

Get user parameters or default parameters if unspecified.

    params = get_params('myparams.prm')  # from a PRM file
    params = get_params(param1=?, ...)  # directly
    param1 = params['param1']

### Create experiment files

    create_files('myexperiment', prm=prm, prb=prb)
    files = open_files('myexperiment', mode='a')
    
### Adding data to an experiment

    with Experiment('myexperiment', mode='a') as exp:
        # Append high-pass filtered data to the experiment.
        exp.recordings[0].high.data.append(filtered)

        # Append a spike.
        exp.spikes.add(time_samples=..., ...)
        
        # Update waveforms of certain spikes.
        exp.spikes.waveforms[indices, ...] = waveforms
