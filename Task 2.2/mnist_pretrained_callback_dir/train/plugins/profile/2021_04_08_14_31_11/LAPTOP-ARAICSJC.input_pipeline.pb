	m????,l@m????,l@!m????,l@	__?Fe??__?Fe??!__?Fe??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$m????,l@??{??P??A??ͪ?'l@Yt$???~??*	??????X@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?U???؟?!ZrBo?I?@)-C??6??1?0?(?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?sF????!???)?{>@)Zd;?O???1??1??#7@:Preprocessing2F
Iterator::Model???<,Ԛ?!?X??[:@)K?=?U??1?}̣??.@:Preprocessing2U
Iterator::Model::ParallelMapV246<?R??!????|?%@)46<?R??1????|?%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip46<???!???iR@)vq?-??1g?:n?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?<,Ԛ?}?!??jR`@)?<,Ԛ?}?1??jR`@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?I+?v?!eF??!@)?I+?v?1eF??!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???????!??[.4A@)?q????o?1I???ic@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9__?Fe??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??{??P????{??P??!??{??P??      ??!       "      ??!       *      ??!       2	??ͪ?'l@??ͪ?'l@!??ͪ?'l@:      ??!       B      ??!       J	t$???~??t$???~??!t$???~??R      ??!       Z	t$???~??t$???~??!t$???~??JCPU_ONLYY__?Fe??b 