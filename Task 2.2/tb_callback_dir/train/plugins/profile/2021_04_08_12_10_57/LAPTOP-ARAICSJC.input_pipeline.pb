	O@aÎ?@O@aÎ?@!O@aÎ?@	???ZR@???ZR@!???ZR@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$O@aÎ?@	??g????A?? ?r?b@Y0*???y@*	?????,A2P
Iterator::Model::Prefetch?):???y@!?@	??X@)?):???y@1?@	??X@:Preprocessing2F
Iterator::Model5?8EG?y@!      Y@)?W[?????1|??ί?}?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 73.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9???ZR@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
		??g????	??g????!	??g????      ??!       "      ??!       *      ??!       2	?? ?r?b@?? ?r?b@!?? ?r?b@:      ??!       B      ??!       J	0*???y@0*???y@!0*???y@R      ??!       Z	0*???y@0*???y@!0*???y@JCPU_ONLYY???ZR@b 