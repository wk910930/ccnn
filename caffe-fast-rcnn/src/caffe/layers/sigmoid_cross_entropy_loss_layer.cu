#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
