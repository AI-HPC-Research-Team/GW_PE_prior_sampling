from gwpy.timeseries import TimeSeries
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

allevents = {} #存放读取的数据
with open("../events/GWOSC_allevents_meta.json",'r',encoding='utf-8') as json_file:
        allevents = json.load(json_file)
        
# BBHs in GWTC1
event_list = ['GW150914', 'GW151012', 'GW151226',
              'GW170104', 'GW170608', 'GW170729',
              'GW170809', 'GW170814', 'GW170818', 'GW170823']


for event in tqdm(event_list):

    event_version = sorted([key for key, value in allevents.items() if event in key.split('-')[0]], 
                           key=lambda x: int(x.split('-')[-1][-1]), reverse=True)[0]
    assert allevents[event_version]['commonName'] == event

    detectors=sorted(list(set([meta['detector'] for meta in allevents[event_version]['strain']])))
    
    t_event = allevents[event_version]['GPS'] # GPS time of coalescence
    print(event, t_event)

    T = 8.0  # number of seconds to analyze in a segment
    T_psd = 1024.0  # number of seconds of data for estimating PSD
    T_buffer = 2.0  # buffer time after the event to include    
    
    try:
        i = 1
        j = 0
        h1_inject = TimeSeries.fetch_open_data('H1', t_event+ T_buffer - T- i*T_psd, t_event+ T_buffer+ j*T_psd, cache=True)
        print('H1:', len(h1_inject))
        assert False == (True in np.isnan(h1_inject.value))
        l1_inject = TimeSeries.fetch_open_data('L1', t_event+ T_buffer - T- i*T_psd, t_event+ T_buffer+ j*T_psd, cache=True)
        print('L1:', len(l1_inject))
        assert False == (True in np.isnan(l1_inject.value))
        
        if len(detectors) == 3:
            v1_inject = TimeSeries.fetch_open_data('V1', t_event+ T_buffer - T- i*T_psd, t_event+ T_buffer+ j*T_psd, cache=True)
            print('V1:', len(v1_inject))
            assert False == (True in np.isnan(v1_inject.value))     

    except:
        i = 0
        j = 1
        h1_inject = TimeSeries.fetch_open_data('H1', t_event+ T_buffer - T- i*T_psd, t_event+ T_buffer+ j*T_psd, cache=True)
        print('H1:', len(h1_inject))
        assert False == (True in np.isnan(h1_inject.value))
        l1_inject = TimeSeries.fetch_open_data('L1', t_event+ T_buffer - T- i*T_psd, t_event+ T_buffer+ j*T_psd, cache=True)
        print('L1:', len(l1_inject))
        assert False == (True in np.isnan(l1_inject.value))
        if len(detectors) == 3:
            v1_inject = TimeSeries.fetch_open_data('V1', t_event+ T_buffer - T- i*T_psd, t_event+ T_buffer+ j*T_psd, cache=True)
            print('V1:', len(v1_inject))
            assert False == (True in np.isnan(v1_inject.value))    

    # Save to file 
    event_dir = Path('./{}'.format(event))
    event_dir.mkdir(parents=True, exist_ok=True)
    
    h1_inject.write(event_dir / 'H-H1_GWOSC_4KHZ-{}-{}.hdf5'.format(h1_inject.t0.value,
                                int(h1_inject.duration.value)) )
    l1_inject.write(event_dir / 'L-L1_GWOSC_4KHZ-{}-{}.hdf5'.format(l1_inject.t0.value,
                                int(l1_inject.duration.value)) )
    if len(detectors) == 3:    
        v1_inject.write(event_dir / 'V-V1_GWOSC_4KHZ-{}-{}.hdf5'.format(v1_inject.t0.value,
                                    int(v1_inject.duration.value)) )
