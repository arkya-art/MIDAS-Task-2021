	|a2U0?h@|a2U0?h@!|a2U0?h@	
fU?a??
fU?a??!
fU?a??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$|a2U0?h@i o?ſ?AS?!?u?h@Y?J?4??*	?????_@2F
Iterator::Model?|гY???!?3?b?`D@)	??g????1kr?
%E?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/n????!?L?ںK<@)???B?i??1??????7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate	?c???!?/j?7$5@)8??d?`??1x???Z?/@:Preprocessing2U
Iterator::Model::ParallelMapV2??0?*??!???u??"@)??0?*??1???u??"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipY?? ޲?!$?8?G?M@)"??u????1.????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice-C??6z?!
???*?@)-C??6z?1
???*?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?I+?v?!?O??T?@)?I+?v?1?O??T?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapu????!??Z?	8@)??H?}m?1???'?&@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9
fU?a??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	i o?ſ?i o?ſ?!i o?ſ?      ??!       "      ??!       *      ??!       2	S?!?u?h@S?!?u?h@!S?!?u?h@:      ??!       B      ??!       J	?J?4???J?4??!?J?4??R      ??!       Z	?J?4???J?4??!?J?4??JCPU_ONLYY
fU?a??b 