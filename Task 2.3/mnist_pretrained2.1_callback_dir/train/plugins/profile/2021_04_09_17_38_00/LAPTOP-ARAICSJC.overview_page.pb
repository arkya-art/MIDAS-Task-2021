?	?3??7?j@?3??7?j@!?3??7?j@	??S?????S???!??S???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?3??7?j@u?V??A?>W[??j@Yd]?Fx??*	???????@2F
Iterator::Model;?O??n??!?%?Ī?U@)?q??????1 +?}A?T@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL7?A`???!x????#@)}??b٭?1?T?y}l!@:Preprocessing2U
Iterator::Model::ParallelMapV2?W[?????!j^?(@)?W[?????1j^?(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateŏ1w-!??!???g?+@)??y?):??1??֫uG??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$????۷?!iо٩?+@)?St$????1=07???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor? ?	??!????h??)? ?	??1????h??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice ?o_?y?!?αF" ??) ?o_?y?1?αF" ??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ݓ??Z??![??@)???_vOn?1??5Y???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??S???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	u?V??u?V??!u?V??      ??!       "      ??!       *      ??!       2	?>W[??j@?>W[??j@!?>W[??j@:      ??!       B      ??!       J	d]?Fx??d]?Fx??!d]?Fx??R      ??!       Z	d]?Fx??d]?Fx??!d]?Fx??JCPU_ONLYY??S???b Y      Y@q?xy??? @"?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 