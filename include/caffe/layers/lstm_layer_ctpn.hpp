#ifndef CAFFE_LSTM_LAYER_1_HPP
#define CAFFE_LSTM_LAYER_1_HPP

#include <string>
#include <vector>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Long-short term memory layer.
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Ftype, typename Btype>
class LstmLayer : public Layer<Ftype, Btype> {
 public:
  explicit LstmLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Lstm"; }
  virtual bool IsRecurrent() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
      NOT_IMPLEMENTED;
  }

  int I_; // input dimension
  int H_; // num of hidden units
  int T_; // length of sequence
  int N_; // batch size
  
  Ftype clipping_threshold_; // threshold for clipped gradient
  shared_ptr<Blob> bias_multiplier_;

  TBlob<Ftype> top_;       // output values
  TBlob<Ftype> cell_;      // memory cell
  TBlob<Ftype> pre_gate_;  // gate values before nonlinearity
  TBlob<Ftype> gate_;      // gate values after nonlinearity

  TBlob<Ftype> c_0_; // previous cell state value
  TBlob<Ftype> h_0_; // previous hidden activation value
  TBlob<Ftype> c_T_; // next cell state value
  TBlob<Ftype> h_T_; // next hidden activation value

  //intermediate values
  TBlob<Ftype> h_to_gate_;
  TBlob<Ftype> h_to_h_;
};

} // namespace caffe

#endif /* ifndef CAFFE_LSTM_LAYER_1_HPP */
