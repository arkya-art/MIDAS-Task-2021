	?A`?Ѐl@?A`?Ѐl@!?A`?Ѐl@	#?Y@???#?Y@???!#?Y@???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?A`?Ѐl@? ?	???A?|?5^vl@Y??h o???*	    @^@2F
Iterator::ModelEGr????!??eP*LC@)P?s???1[=;n,=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ZӼ???!??zv?@@)e?X???1?ˠT?<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?0?*???!(&ޏ?0@)%u???1??eP*L(@:Preprocessing2U
Iterator::Model::ParallelMapV2?+e?X??!L?9??"@)?+e?X??1L?9??"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??&S??!?r??ճN@)U???N@??1pc?
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??+ey?!?A?0?~@)a??+ey?1?A?0?~@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??_vOv?! 9????@)??_vOv?1 9????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap=?U?????!?9???3@)????Mbp?1?T?x?r
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9#?Y@???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? ?	???? ?	???!? ?	???      ??!       "      ??!       *      ??!       2	?|?5^vl@?|?5^vl@!?|?5^vl@:      ??!       B      ??!       J	??h o?????h o???!??h o???R      ??!       Z	??h o?????h o???!??h o???JCPU_ONLYY#?Y@???b 