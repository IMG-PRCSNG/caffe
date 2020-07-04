#include "caffe/layers/reverse_layer.hpp"

namespace caffe {
template <typename Dtype>
void reverse_cpu(const int count, const Dtype* from_data, Dtype* to_data, 
    const int* counts, const int axis_count, const int axis) {
    for(int index=0; index<count; index++) {
        int ind=(index/counts[axis])%axis_count;
        int to_index=counts[axis]*(axis_count-2*ind-1)+index;
        *(to_data+to_index)=*(from_data+index);
    }
}
template <typename Ftype, typename Btype>
void ReverseLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
        const vector<Blob*>& top) {
    CHECK_NE(bottom[0], top[0])<<this->type()<<" does not support in-place computation.";
    reverse_param_=this->layer_param_.reverse_param();
}

template <typename Ftype, typename Btype>
void ReverseLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
    vector<int> shape=bottom[0]->shape();
    axis_=reverse_param_.axis();
    CHECK_GT(shape.size(), 0)<<this->type()<<" does not support 0 axes blob.";
    CHECK_GE(axis_, 0)<<"axis must be greater than or equal to 0.";
    CHECK_LT(axis_, shape.size())<<"axis must be less than bottom's dimension.";
    top[0]->ReshapeLike(*bottom[0]);
    const int dim=shape.size();
    shape.clear();
    shape.push_back(dim);
    bottom_counts_.Reshape(shape);
    int* p=bottom_counts_.mutable_cpu_data();
    for (int i=1; i<dim; i++) {
        *p=bottom[0]->count(i);
        p++;
    }
    *p=1;
}

template <typename Ftype, typename Btype>
void ReverseLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom, 
        const vector<Blob*>& top) {
    reverse_cpu<Ftype>(bottom[0]->count(), bottom[0]->cpu_data<Ftype>(), 
        top[0]->mutable_cpu_data<Ftype>(), bottom_counts_.cpu_data(), 
        bottom[0]->shape(axis_), axis_);
}

#ifdef CPU_ONLY
STUB_GPU(ReverseLayer);
#endif

INSTANTIATE_CLASS_FB(ReverseLayer);
REGISTER_LAYER_CLASS(Reverse);

}  // namespace caffe
