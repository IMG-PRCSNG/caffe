#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/lstm_layer_ctpn.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Ftype, typename Btype>
void LstmLayer<Ftype,Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  clipping_threshold_ = this->layer_param_.lstm_param().clipping_threshold();
  N_ = bottom[0]->shape(1); // batch_size
  H_ = this->layer_param_.lstm_param().num_output(); // number of hidden units
  I_ = bottom[0]->count(2); // input dimension

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    shared_ptr<Filler<Ftype> > weight_filler(GetFiller<Ftype>(
        this->layer_param_.lstm_param().weight_filler()));
 
    // input-to-hidden weights
    // Intialize the weight
    vector<int> weight_shape;
    weight_shape.push_back(4*H_);
    weight_shape.push_back(I_);
    this->blobs_[0].reset(new TBlob<Ftype>(weight_shape));
    weight_filler->Fill(this->blobs_[0].get());

    // hidden-to-hidden weights
    // Intialize the weight
    weight_shape.clear();
    weight_shape.push_back(4*H_);
    weight_shape.push_back(H_);
    this->blobs_[1].reset(new TBlob<Ftype>(weight_shape));
    weight_filler->Fill(this->blobs_[1].get());

    // If necessary, intiialize and fill the bias term
    vector<int> bias_shape(1, 4*H_);
    this->blobs_[2].reset(new TBlob<Ftype>(bias_shape));
    shared_ptr<Filler<Ftype> > bias_filler(GetFiller<Ftype>(
        this->layer_param_.lstm_param().bias_filler()));
    bias_filler->Fill(this->blobs_[2].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  
}

template <typename Ftype, typename Btype>
void LstmLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  // Figure out the dimensions
  T_ = bottom[0]->shape(0); // length of sequence
  N_=bottom[0]->shape(1);
  CHECK_EQ(bottom[0]->count(2), I_) << "Input size "
    "incompatible with inner product parameters.";
  vector<int> original_top_shape;
  original_top_shape.push_back(T_);
  original_top_shape.push_back(N_);
  original_top_shape.push_back(H_);
  top[0]->Reshape(original_top_shape);

  vector<int> cell_shape;
  cell_shape.push_back(N_);
  cell_shape.push_back(H_);
  c_0_.Reshape(cell_shape);
  h_0_.Reshape(cell_shape);
  c_T_.Reshape(cell_shape);
  h_T_.Reshape(cell_shape);
  h_to_h_.Reshape(cell_shape);

  vector<int> gate_shape;
  gate_shape.push_back(N_);
  gate_shape.push_back(4);
  gate_shape.push_back(H_);
  h_to_gate_.Reshape(gate_shape);

  // Gate initialization
  gate_shape.clear();
  gate_shape.push_back(T_);
  gate_shape.push_back(N_);
  gate_shape.push_back(4);
  gate_shape.push_back(H_);
  pre_gate_.Reshape(gate_shape);
  gate_.Reshape(gate_shape);
  
  vector<int> top_shape;
  top_shape.push_back(T_);
  top_shape.push_back(N_);
  top_shape.push_back(H_);
  cell_.Reshape(top_shape);
  top_.Reshape(top_shape);
  top_.ShareData(*top[0]);
  top_.ShareDiff(*top[0]);
  
  // Set up the bias multiplier
  vector<int> multiplier_shape(1, N_*T_);
  bias_multiplier_.reset(new TBlob<Ftype>(multiplier_shape));
  caffe_set(bias_multiplier_->count(), Ftype(1), 
    bias_multiplier_->mutable_cpu_data<Ftype>());
}

template <typename Ftype, typename Btype>
void LstmLayer<Ftype, Btype>::Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  CHECK_EQ(top[0]->cpu_data<Ftype>(), top_.cpu_data());
  Ftype* top_data = top_.mutable_cpu_data();
  const Ftype* bottom_data = bottom[0]->cpu_data<Ftype>();
  const Ftype* clip = NULL;
  if (bottom.size() > 1) {
    clip = bottom[1]->cpu_data<Ftype>();
    CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  }
  const Ftype* weight_i = this->blobs_[0]->template cpu_data<Ftype>();
  const Ftype* weight_h = this->blobs_[1]->template cpu_data<Ftype>();
  const Ftype* bias = this->blobs_[2]->template cpu_data<Ftype>();
  Ftype* pre_gate_data = pre_gate_.mutable_cpu_data();
  Ftype* gate_data = gate_.mutable_cpu_data();
  Ftype* cell_data = cell_.mutable_cpu_data();
  Ftype* h_to_gate = h_to_gate_.mutable_cpu_data();

  // Initialize previous state
  if (clip) {
    caffe_copy(c_0_.count(), c_T_.cpu_data(), c_0_.mutable_cpu_data());
    caffe_copy(h_0_.count(), h_T_.cpu_data(), h_0_.mutable_cpu_data());
  }
  else {
    caffe_set(c_0_.count(), Ftype(0.), c_0_.mutable_cpu_data());
    caffe_set(h_0_.count(), Ftype(0.), h_0_.mutable_cpu_data());
  }

  // Compute input to hidden forward propagation
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, T_*N_, 4*H_, I_, Ftype(1.),
      bottom_data, weight_i, Ftype(0.), pre_gate_data);
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, T_*N_, 4*H_, 1, Ftype(1.),
      bias_multiplier_->cpu_data<Ftype>(), bias, Ftype(1.), pre_gate_data);

  // Compute recurrent forward propagation
  for (int t = 0; t < T_; ++t) {
    Ftype* h_t = top_data + top_.offset(t);
    Ftype* c_t = cell_data + cell_.offset(t);
    Ftype* pre_gate_t = pre_gate_data + pre_gate_.offset(t);
    Ftype* gate_t = gate_data + gate_.offset(t);
    Ftype* h_to_gate_t = h_to_gate;
    const Ftype* clip_t = clip ? clip + bottom[1]->offset(t) : NULL;
    const Ftype* h_t_1 = t > 0 ? (h_t - top_.offset(1)) : h_0_.cpu_data();
    const Ftype* c_t_1 = t > 0 ? (c_t - cell_.offset(1)) : c_0_.cpu_data();

    // Hidden-to-hidden propagation
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, N_, 4*H_, H_, Ftype(1.), 
        h_t_1, weight_h, Ftype(0.), h_to_gate);

    for (int n = 0; n < N_; ++n) {
      const bool cont = clip_t ? (bool)(clip_t[n]) : t > 0;
      if (cont) {
        caffe_add(4*H_, pre_gate_t, h_to_gate, pre_gate_t);
      }
      for (int d = 0; d < H_; ++d) {
        // Apply nonlinearity
        gate_t[d] = sigmoid(pre_gate_t[d]);
        gate_t[H_ + d] = cont ? sigmoid(pre_gate_t[H_ + d]) : Ftype(0.);
        gate_t[2*H_ + d] = sigmoid(pre_gate_t[2*H_ + d]);
        gate_t[3*H_ + d] = tanh(pre_gate_t[3*H_ + d]);

        // Compute cell : c(t) = f(t)*c(t-1) + i(t)*g(t)
        c_t[d] = gate_t[H_ + d] * c_t_1[d] + gate_t[d] * gate_t[3*H_ + d];
        h_t[d] = gate_t[2*H_ + d] * tanh(c_t[d]);
      }
      
      h_t += H_;
      c_t += H_;
      c_t_1 += H_;
      pre_gate_t += 4*H_;
      gate_t += 4*H_;
      h_to_gate_t += 4*H_;
    }
  }
  // Preserve cell state and output value for truncated BPTT
  caffe_copy(N_*H_, cell_data + cell_.offset(T_-1), c_T_.mutable_cpu_data());
  caffe_copy(N_*H_, top_data + top_.offset(T_-1), h_T_.mutable_cpu_data());
}


#ifdef CPU_ONLY
STUB_GPU(LstmLayer);
#endif

INSTANTIATE_CLASS_FB(LstmLayer);
REGISTER_LAYER_CLASS(Lstm);

}  // namespace caffe
