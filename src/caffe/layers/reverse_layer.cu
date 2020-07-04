#include "caffe/layers/reverse_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void reverse_gpu(const int nthreads, const Dtype* from_data, Dtype* to_data, 
	const int* counts, const int axis_count, const int axis) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  	int ind=(index/counts[axis])%axis_count;
  	int to_index=counts[axis]*(axis_count-2*ind-1)+index;
  	*(to_data+to_index)=*(from_data+index);
  }
}

template <typename Ftype, typename Btype>
void ReverseLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom, 
		const vector<Blob*>& top) {
	const int nthreads=bottom[0]->count();
	reverse_gpu<Ftype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom[0]->gpu_data<Ftype>(), top[0]->mutable_gpu_data<Ftype>(), 
        bottom_counts_.gpu_data(), bottom[0]->shape(axis_), axis_);
}



INSTANTIATE_LAYER_GPU_FORWARD_ONLY_FB(ReverseLayer);

}  // namespace caffe
