	2U0*??{@2U0*??{@!2U0*??{@	?G???~R@?G???~R@!?G???~R@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$2U0*??{@r??????A?A?f՗\@Y???oZt@*	????Z?A2P
Iterator::Model::PrefetchjM?Xt@!?E?~b?X@)jM?Xt@1?E?~b?X@:Preprocessing2F
Iterator::Model??ׁsYt@!      Y@)Ǻ????1eE?"???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 74.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?G???~R@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	r??????r??????!r??????      ??!       "      ??!       *      ??!       2	?A?f՗\@?A?f՗\@!?A?f՗\@:      ??!       B      ??!       J	???oZt@???oZt@!???oZt@R      ??!       Z	???oZt@???oZt@!???oZt@JCPU_ONLYY?G???~R@b 