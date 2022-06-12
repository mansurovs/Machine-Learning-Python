# We display all streamer parameters with their default values.
# See documentation for detailed information about each parameter.
# https://www.nfstream.org/docs/api#nfstreamer

from nfstream import NFStreamer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
#if __name__ == '__main__':
# df = NFStreamer(source="720p@30Hzdemand.pcap").to_pandas()
# X = df[["bidirectional_packets", "bidirectional_bytes","application_name","application_category_name"]]
# y = df["application_category_name"]
# Index = pd.Index(['Web','Network','Cloud','Media','System','SoftwareUpdate'])
# data = np.array([0,1,2,3,4,5])
# y1= pd.DataFrame({'num':data},index=Index)
# y2 = pd.DataFrame({'object':y}, index= Index)
# model = RandomForestClassifier().fit(X, y)


my_streamer = NFStreamer(source="Youtube_720p_50fps.pcap",  # or network interface (source="eth0")
                         decode_tunnels=True,
                         bpf_filter=None,
                         promiscuous_mode=True,
                         snapshot_length=1536,
                         idle_timeout=120,
                         active_timeout=1800,
                         accounting_mode=2,
                         udps=None,
                         n_dissections=20,
                         statistical_analysis=True,
                         splt_analysis=0,
                         n_meters=0,
                         performance_report=5,
                         system_visibility_mode=0,
                         system_visibility_poll_ms=100,
                         system_visibility_extension_port=28314,
                         )

if __name__ == '__main__':
    for flow in my_streamer:
        print(flow)  # print it.
        #    print(flow.to_namedtuple()) # convert it to a namedtuple.
        #    print(flow.to_json()) # convert it to json.
        print(flow.keys())  # get flow keys.
        print(flow.values())  # get flow values.

        my_dataframe = my_streamer.to_pandas(columns_to_anonymize=[])
        my_dataframe.head()
        total_flows_count = my_streamer.to_csv(path=None, columns_to_anonymize=[], flows_per_file=0)