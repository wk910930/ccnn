#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithCascadeLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
//   LOG(INFO)<<"zx, SoftmaxWithCascadeLossLayer: top.size() = "<<top.size();
//   LossLayer<Dtype>::LayerSetUp(bottom, top);
//   LOG(INFO)<<"zx, after LayerParameter";
//   LayerParameter softmax_param(this->layer_param_);
//   softmax_param.set_type("Softmax");
//   softmax_param.clear_loss_weight();
//   softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
//   softmax_bottom_vec_.clear();
//   softmax_bottom_vec_.push_back(bottom[0]);
//   softmax_top_vec_.clear();
//   softmax_top_vec_.push_back(&prob_);
//   LOG(INFO)<<"zx, before softmax_layer_ setup";
//   softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
//   LOG(INFO)<<"zx, after softmax_layer_ setup";
// 
//   has_ignore_label_ =
//     this->layer_param_.loss_param().has_ignore_label();
//   if (has_ignore_label_) {
//     ignore_label_ = this->layer_param_.loss_param().ignore_label();
//   }
//   normalize_ = this->layer_param_.loss_param().normalize();
  iter_cnt_ = 0;
  PosCnt_=0;
  NegCnt_=0;
  BPCnt_=0;
  Pos_scores_=0;
  Neg_scores_=0;
  Step3_Cnt = 0;
  Ratio_keep_Cnt =0;
  LayerParameter softmax_param(this->layer_param_);
  LOG(INFO)<<"zx, after LayerParameter";
  softmax_param.set_type("Softmax");
  LOG(INFO)<<"zx, before LayerRegistry<Dtype>::CreateLayer";
  softmax_param.clear_loss_weight();
  softmax_param.clear_loss_param();
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  // zx, bottom[0]: cls_score, bottom[1]: labels
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  LOG(INFO)<<"zx, before softmax_layer_ setup";
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  int inner_num2_ = bottom[0]->count(softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  LOG(INFO) << "outer_num_: " << outer_num_ << " inner_num_: "<< inner_num_ << " inner_num2_: " << inner_num2_;
  
  LOG(INFO)<<"zx, after softmax_layer_ setup";
  
  LOG(INFO)<<"zx, SoftmaxWithCascadeLossLayer: top.size() = "<<top.size();
  // zx, hard, begin
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
    this->layer_param_.add_loss_weight(Dtype(0));
  }
  // zx, hard, end
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  // zx, has_ignore_label_ == false
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  LOG(INFO)<<"ignore_label = "<<ignore_label_;

  // zx, normalize default value is true.
  normalize_ = this->layer_param_.loss_param().normalize();
  LOG(INFO)<<"normalize = "<<normalize_;
  
  keep_ratio_ = this->layer_param_.loss_param().keep_ratio();
  if (keep_ratio_ < 0.1)
      keep_ratio_ = 0.1;

  cascade_type_ = this->layer_param_.loss_param().cascade_type();
  rand_pos_ = this->layer_param_.loss_param().rand_pos();
  nms_threshold_ = this->layer_param_.loss_param().nms_threshold();
  // zx, cascade, begin, add "threshold_"
  //threshold_ = this->layer_param_.softmax_param().threshold();
  threshold_ = this->layer_param_.loss_param().threshold();
  rand_ratio_ = this->layer_param_.loss_param().rand_ratio();

  bp_size_ = this->layer_param_.loss_param().bp_size();
  LOG(INFO)<<"bp_size = "<<bp_size_;

  hard_mining_ = this->layer_param_.loss_param().hard_mining();
  LOG(INFO)<<"hard_mining = "<<hard_mining_;

  sampling_ = this->layer_param_.loss_param().sampling();
  LOG(INFO)<<"sampling = "<<sampling_;

  cascade_ = this->layer_param_.loss_param().cascade();
  LOG(INFO)<<"cascade = "<<cascade_;

  if (this->layer_param_.loss_param().has_batch_size()) {
    batch_size_ = this->layer_param_.loss_param().batch_size();
    LOG(INFO)<<"batch_size = "<<batch_size_;
  }

  if (this->layer_param_.loss_param().has_gt_batch_size()) {
    gt_batch_size_ = this->layer_param_.loss_param().gt_batch_size();
    LOG(INFO)<<"gt_batch_size = "<<gt_batch_size_;
  }

  if (this->layer_param_.loss_param().has_ims_per_batch()) {
    ims_per_batch_ = this->layer_param_.loss_param().ims_per_batch();
    LOG(INFO)<<"ims_per_batch = "<<ims_per_batch_;
  }

  if (this->layer_param_.loss_param().has_gt_per_batch()) {
    gt_per_batch_ = this->layer_param_.loss_param().gt_per_batch();
    LOG(INFO)<<"gt_per_batch = "<<gt_per_batch_;
  }

  if (this->layer_param_.loss_param().has_fg_fraction()) {
    fg_fraction_ = this->layer_param_.loss_param().fg_fraction();
    LOG(INFO)<<"fg_fraction = "<<fg_fraction_;
  }


  // zx, cascade, end

  //LOG(INFO)<<"has_ignore_label_ = "<<has_ignore_label_<<"normalize_ = "<<normalize_;
  //LOG(INFO)<<"bottom.size() = "<<bottom.size();

  // zx, bottom.size() == 2
  CHECK_EQ(bottom[1]->channels(), 1);
  if (bottom.size() == 4)
  {
    CHECK_EQ(bottom[1]->num(), bottom[2]->num());
    CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[1]->height(), bottom[2]->height());
    CHECK_EQ(bottom[1]->width(), bottom[2]->width());
  }
  
}

template <typename Dtype>
void SoftmaxWithCascadeLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
//  LOG(INFO) << "Original outer_num_: " << outer_num_ << " inner_num_: " << inner_num_;
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  foot_print_.resize(outer_num_*inner_num_);
//  LOG(INFO) << "outer_num_: " << outer_num_ << " inner_num_: " << inner_num_;
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[1]);
  }

}

template <typename Dtype>
void SoftmaxWithCascadeLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LOG(INFO)<<"enter SoftmaxWithCascadeLossLayer<Dtype>::Forward_cpu";
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  bool has_weight = bottom.size() >= 3;
  // zx, has_weight == 0
  const Dtype* weight;
  if (has_weight)
    weight = bottom[2]->cpu_data();
  int dim = prob_.count() / outer_num_;
  Dtype weights_sum = Dtype(0.0);
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {

      const Dtype weight_value = has_weight ? static_cast<Dtype>(weight[i * inner_num_ + j]) : 1;
      // zx, weight_value == 1
      DCHECK_GE(weight_value, 0);

      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= weight_value * log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      weights_sum += weight_value;
    }
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = weights_sum == Dtype(0.0) ? Dtype(0.0) : (loss / weights_sum);
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithCascadeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    bool has_weight = bottom.size() >= 3;
    const Dtype* weight;
    if (has_weight)
        weight = bottom[2]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int channels = prob_.channels();
    Dtype weights_sum = Dtype(0.0);
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;

          if (has_weight){
            const Dtype weight_value = static_cast<Dtype>(weight[i * inner_num_ + j]);
            for (int k = 0; k < channels; ++k)
            bottom_diff[i * dim + k * inner_num_ + j] *= weight_value;

            weights_sum += weight_value;
          }
          else{
            weights_sum += 1.0;
          }
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), weights_sum == Dtype(0.0) ? Dtype(0.0) : (loss_weight / weights_sum), bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithCascadeLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithCascadeLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithCascadeLoss);

}  // namespace caffe
