#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template <typename Dtype>  void nms( const Dtype* bottom_rois, const Dtype* prob_data_cpu, const Dtype* labels, int* bp_map_cpu, float thresh, int count, int dim )
{
    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::vector<std::pair<Dtype, int> > fw_positive_map;
    std::vector<std::pair<Dtype, int> > fw_negative_map;

    std::vector<Dtype> area(count);
//     LOG(INFO)<< "Entered...";
//     printf("thresh: %.3f, count: %d, dim: %d\n", thresh, count, dim);
	for (size_t i = 0; i < count; ++i)
	{
        std::pair<Dtype, int> fw_sample;
        const int label_value = static_cast<int>(labels[i]);        
//         printf("iter %d, label_value: %d\n", i, label_value);
        fw_sample = std::make_pair(prob_data_cpu[i*dim + label_value], i);
        if (label_value==0)
            fw_negative_map.push_back(fw_sample);
        else
            fw_positive_map.push_back(fw_sample);
        Dtype x1i = (bottom_rois[i*5+1]);
        Dtype y1i = (bottom_rois[i*5+2]);
        Dtype x2i = (bottom_rois[i*5+3]);
        Dtype y2i = (bottom_rois[i*5+4]);
        area[i] = (x2i-x1i+1) * (y2i-y1i+1);
	}
//     LOG(INFO)<< "sorting...";
    std::sort(fw_positive_map.begin(), fw_positive_map.end());
    std::sort(fw_negative_map.begin(), fw_negative_map.end());
    for (int p=0; p<2; p++)
    {
        int i;
        int inc = 1;
        if (p==1)
        {
            count = fw_positive_map.size();
            i = count-1;
            inc = -1;
        }
        else
        {
            count = fw_negative_map.size();
            i = 0;
            inc = 1;
        }
//         LOG(INFO)<< "NMS...";
        for (; (i>=0)&&(i<count); i+=inc)
        // keep looping while some indexes still remain in the indexes list
        {
            int id, j;
            if (p==1)
            {
                id = fw_positive_map[i].second;
                j = i-1;
            }
            else
            {
                id = fw_negative_map[i].second;
                j = i+1;
            }
//         if (i<=0)
//             LOG(INFO)<< "NMS...";
//         printf("iter %d: %d\n", i, id);
            if (bp_map_cpu[id])
            {
                int roi_batch_ind = bottom_rois[id*5+0];
                Dtype x1i = (bottom_rois[id*5+1]);
                Dtype y1i = (bottom_rois[id*5+2]);
                Dtype x2i = (bottom_rois[id*5+3]);
                Dtype y2i = (bottom_rois[id*5+4]);
                for (; (j>=0)&&(j<i); j+=inc)
                {
                    int id2;
                    if (p==1)
                        id2 = fw_positive_map[j].second;
                    else
                        id2 = fw_negative_map[j].second;
//                 printf("iter (%d, %d): (%d, %d)\n", i, j, id, id2);
                    if ( bp_map_cpu[id2] && (bottom_rois[id2*5+0] == roi_batch_ind) )
                    {
                        bool Examin_flag = (labels[id2]==labels[id]) || ((labels[id]>0)&&(labels[id2]==0));
                        if (Examin_flag)
                        {
                            Dtype x1j = (bottom_rois[id2*5+1]);
                            Dtype y1j = (bottom_rois[id2*5+2]);
                            Dtype x2j = (bottom_rois[id2*5+3]);
                            Dtype y2j = (bottom_rois[id2*5+4]);
                            Dtype xx1 = max(x1i, x1j);
                            Dtype yy1 = max(y1i, y1j);
                            Dtype xx2 = min(x2i, x2j);
                            Dtype yy2 = min(y2i, y2j);
                            Dtype w = max((Dtype)0, xx2-xx1+1);
                            Dtype h = max((Dtype)0, yy2-yy1+1);
                            Dtype inter = w*h;
                            Dtype ovr;
                            if (w > 0 && h > 0)
                                ovr = inter / (area[id] + area[id2] - inter);
                            else
                                ovr = 0;
                            if (ovr > thresh)
                            {
                                bp_map_cpu[id2] = 0;
//                                 LOG(INFO)<<id2 <<": ovr: " << ovr << "thresh: " << thresh << " " << id << " killed" << id2;
                            }
                        }
                    }
                }
            }
        }
    }
}
    
    
    template <typename Dtype>
            __global__ void SoftmaxLossForwardGPU2(const int nthreads, Dtype* prob_data, const Dtype* prob_map, const Dtype* label, const Dtype *weight, Dtype* loss,
            const int num, const int dim, const int spatial_dim,
            const bool has_ignore_label_, const bool hard_mining, const int ignore_label_,
            Dtype* counts, const Dtype* bp_map) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            // zx,   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
            
            // zx, cascade, begin.
            if (hard_mining) {
                loss[index] = 0;
                counts[index] = 0;
            }
            // zx, cascade, end.
            
            // zx, nthreads == 48, spatial_dim == 1, label.count() == 48, dim == 201
            const int n = index / spatial_dim;
            const int s = index % spatial_dim;
            const int label_value = static_cast<int>(label[n * spatial_dim + s]);
            // zx, has_ignore_label_ == false
            if (has_ignore_label_ && label_value == ignore_label_) {
                loss[index] = 0;
                counts[index] = 0;
            } else {
                // zx, weight == NULL, weight_value == 1
                const Dtype weight_value = (weight != NULL) ? static_cast<Dtype>(weight[n * spatial_dim + s]) : 1;
                // zx, hard, begin
                if (hard_mining && bp_map) {
                    if (bp_map[n] > 0 && prob_data) {
                        counts[index] = weight_value;
                        loss[index] = - weight_value * log(max(prob_data[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN)));
                    }
                }
                // zx, hard, end
                else {
                    // zx, original implementation.
                    loss[index] = - weight_value * log(max(prob_data[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN)));
                    counts[index] = weight_value;
                }
            }
        }
    }
    
    template <typename Dtype>
            void SoftmaxWithCascadeLossLayer<Dtype>::Forward_gpu(
            const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        iter_cnt_++;
//         LOG(INFO) << "Forward";
        //printf("Forward");
  int softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  int inner_num2_ = bottom[0]->count(softmax_axis_);
//  LOG(INFO)  << "inner_num2_:" << inner_num2_;
        const int nthreads = outer_num_ * inner_num_;
//        LOG(INFO)  << "outer_num_:" << outer_num_<< " inner_num_:" << inner_num_ << " nthreads:" << nthreads;
//        LOG(INFO) << "bottom[1]->count: " << bottom[1]->count();
//         bottom[1]->mutable_cpu_data();
//         const Dtype* label1 = bottom[1]->cpu_data();
//            for (int index = 0; index < nthreads; index++) {
//                 const Dtype label_value = static_cast<Dtype>(label1[index]);
//                 LOG(INFO) << "label in " << index <<" : " << label_value;
//             }
        softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
        bottom[1]->mutable_gpu_data();
        // zx, prob_.count() = 9648, label.count() = 48
        const Dtype* label = bottom[1]->gpu_data();
        
        // zx, cascade
        // original implementation
        //const Dtype* prob_data = prob_.gpu_data();
        // original implementation
        
        // zx, cascade
        
        // zx, bottom.size() == 2. has_weight == 0
        // zx, hard, begin
        // zx, original implementation, begin
#if 0
        bool has_weight = bottom.size() >= 4;
        const Dtype* weight = NULL;
        if (has_weight)
            weight = bottom[3]->gpu_data();
#endif
        // zx, original implementation, end
        // zx, hard, end
        bool has_weight = bottom.size() >= 4;
        const Dtype* weight = NULL;
//         if (has_weight)
//             weight = bottom[3]->gpu_data();
        
        // zx, outer_num_ == 48(2*16+2*8=48), inner_num_ == 1, prob_.count() = 48*201 = 9648, dim = 9648/48=201
        // zx, outer_num_ == 64(1*32+4(gt_per_batch default: 4)*8=64), inner_num_ = 1, prob_.count() = 64*201 = 12864, dim = 201, label.count() = 64
        // zx, 4 gpus setting, outer_num_ = 64*4=256, inner_num_ == 1, prob_.count() = 4*64*201 = 51456
        const int dim = prob_.count() / outer_num_;
        //LOG(INFO)<<"dim = "<<dim<<", inner_num_ = "<<inner_num_<<", outer_num_ = "<<outer_num_;
        
        // Since this memory is not used for anything until it is overwritten
        // on the backward pass, we use it here to avoid having to allocate new GPU
        // memory to accumulate intermediate results in the kernel.
        Dtype* loss_data = bottom[0]->mutable_gpu_diff();
        
        // Similarly, this memory is never used elsewhere, and thus we can use it
        // to avoid having to allocate additional GPU memory.
        Dtype* counts = prob_.mutable_gpu_diff();
        
        // zx, cascade, begin
        //const int* bp_map_gpu = NULL;
        shared_ptr<Blob<int> > bp_map(new Blob<int>());
        shared_ptr<Blob<Dtype> > prob_map(new Blob<Dtype>());
//        LOG(INFO)  << "Reshape:" << inner_num2_;
        //printf("Reshape");
        bp_map->Reshape(nthreads, 1, 1, 1);
        prob_map->Reshape(nthreads, 2, 1, 1);
        int negative = 0, positive = 0;
        if (  1) {
//         if (  Caffe::MPI_my_rank() == 0) {
            // hard
            const Dtype* prob_data_cpu;
            prob_.mutable_cpu_data();
            prob_data_cpu = prob_.cpu_data();
            
//           // cascade
//           if (cascade_) {
//             prob_data_cpu = bottom[2]->cpu_data();
//           }
            /*
         (hard + sample) on per gpu, total_gpu: 4, bp_size: ((8+24)*1+(2+6)*4) * 4 = 256
         ------------------------------------- --------------------------    -------------------------------------
         | + x%  |     |300 |     |          |                               | + 25% | 8   | 32 |     |          |
         --------------|    |     |          |     hard + sample             --------------|    |     |          |
         |       |     |    | val |          |     ------------------->      |       | 24  |    | val |          |
         | - y%  |     |    |     |          |                               | - 75% |     |    |     |          |
         |       |     |    |     |          |                               |       |     |    |     |          |
         --------------------------          | --------------------------    --------------------------          |
         | + 25% | 2 | 8 |    |              |                               | + 25% | 2 | 8 |    |              |
         ------------|   |    |              |                               ------------|   |    |              |
         |       | 6 |   |    |              |                               |       | 6 |   |    |              |
         | - 75% |   |   |    |              |                               | - 75% |   |   |    |              |
         |       |   |   |    |              |                               |       |   |   |    |              |
         -----------------    |              |                               -----------------    |              |
         | + 25% | 2 | 8 |    |              |                               | + 25% | 2 | 8 |    |              |
         ------------|   |    |              |                               ------------|   |    |              |
         |       | 6 |   |    |              |                               |       | 6 |   |    |              |
         | - 75% |   |   |    | fw_size: 332 |                               | - 75% |   |   |    | bp_size/4:64 |
         |       |   |   |    |              |             keep              |       |   |   |    |              |
         ----------------- gt |              |      ------------------->     ----------------- gt |              |
         | + 25% | 2 | 8 |    |              |                               | + 25% | 2 | 8 |    |              |
         ------------|   |    |              |                               ------------|   |    |              |
         |       | 6 |   |    |              |                               |       | 6 |   |    |              |
         | - 75% |   |   |    |              |                               | - 75% |   |   |    |              |
         |       |   |   |    |              |                               |       |   |   |    |              |
         -----------------    |              |                               -----------------    |              |
         | + 25% | 2 | 8 |    |              |                               | + 25% | 2 | 8 |    |              |
         ------------|   |    |              |                               ------------|   |    |              |
         |       | 6 |   |    |              |                               |       | 6 |   |    |              |
         | - 75% |   |   |    |              |                               | - 75% |   |   |    |              |
         |       |   |   |    |              |                               |       |   |   |    |              |
         ------------------------------------- --------------------------    -------------------------------------
             */
            
            int total_rois_per_gpu = batch_size_ * ims_per_batch_ + gt_batch_size_ * gt_per_batch_;
            int total_gpu_num = nthreads/total_rois_per_gpu;
            int init_val_index_per_gpu;
            int total_val_per_gpu = batch_size_ * ims_per_batch_;
            int total_gt_per_gpu = gt_batch_size_ * gt_per_batch_;
           // printf("176: batch_size_: %d, ims_per_batch_: %d, gt_batch_size_: %d, gt_per_batch_: %d, nthreads: %d, bp_size_: %d\n", batch_size_, ims_per_batch_, gt_batch_size_, gt_per_batch_, nthreads, bp_size_);
            
            //OY
            int Samples_in_loop = total_val_per_gpu + total_gt_per_gpu;
            int Sample_num_per_gpu = bp_size_/total_gpu_num;
            int Sample_num_positive_per_gpu = Sample_num_per_gpu * fg_fraction_;
            int Sample_num_negative_per_gpu = Sample_num_per_gpu - Sample_num_positive_per_gpu;
            // printf("Sample_num_positive_per_gpu: %d, Sample_num_negative_per_gpu: %d \n", Sample_num_positive_per_gpu, Sample_num_negative_per_gpu);
            //OY
            
            std::vector<std::vector<int> > p_n_sample_num_per_gpu(total_gpu_num, std::vector<int>(2));
            
            int* bp_map_cpu = bp_map->mutable_cpu_data();
            //printf("cascade_ = %d \n", cascade_);

            //const Dtype* bp_map_cpu2 = bottom[2]->cpu_data();
            
            Dtype* prob_data_cpu_m = prob_.mutable_cpu_data();            
//             for (int index = 0; index < nthreads; index++) {
//                 for (int ind_cls = 0; ind_cls < 201; ind_cls++) {
//                     LOG(INFO) << "prob (" << index << "," << ind_cls << " :" << prob_data_cpu_m[index * dim + ind_cls];
//                 }
//             }
//             for (int index = 0; index < nthreads; index++) {
//                 const int label_value = static_cast<int>(bottom[1]->cpu_data()[index]);
//                 LOG(INFO) << "label " << index <<" : " << label_value;
//             }

            int bp_cnt_in=0;
          if (!cascade_)
            {
                //printf("not cascade");
                //LOG(INFO) << "not cascade";
                for (int i = 0; i < nthreads; i++) {
                    bp_map_cpu[i] = 1;
                }
                //printf("nthreds: %d\n", nthreads);
            }
            else
            {
                // filter non-bp samples according to bp_map_cpu
//                 LOG(INFO) << "cascade";

                const Dtype* bp_map_cpu2;
                if (nms_threshold_ > 0)
                {
                   bp_map_cpu2 = bottom[bottom.size()-2]->cpu_data();
                   CHECK_EQ(bottom.size(), 4); // Check your prototxt! If use NMS threshold, then we should have 4 bottom inputs: score, label, bp_map and roi
                }
                else
                {
                   bp_map_cpu2 = bottom[bottom.size()-1]->cpu_data();
                   CHECK_EQ(bottom.size(), 3); //  Check your prototxt!  We should have 3 bottom inputs: score, label, and bp_map
                }
                int bp_count = 0;
                for (int index = 0; index < nthreads; index++) {
//                     printf("%d, bp_map in %d:  %d\n",  Caffe::MPI_my_rank(), index, static_cast<int>(bp_map_cpu2[index]));
//                     LOG(INFO) << Caffe::MPI_my_rank() << "bp_map in " << index <<" : " << bp_map_cpu2[index];
                    bp_map_cpu[index] = static_cast<int>(bp_map_cpu2[index]);
                    bp_count += bp_map_cpu[index];
                }
//                 for (int index = 0; index < nthreads; index++) {
//                     const int label_value = static_cast<int>(bottom[1]->cpu_data()[index]);
//                     LOG(INFO) << "label " << index <<" : " << label_value;
//                 }
//                 LOG(INFO)<<"bp_count in: " << bp_count << ", nthreads: " << nthreads;
                bp_cnt_in = bp_count;
                CHECK_GE(nthreads, bp_cnt_in);
                Dtype* prob_data_cpu_m = prob_.mutable_cpu_data();
                    
//                 for (int index = 0; index < nthreads; index++) {
//                     const int label_value = static_cast<int>(bottom[1]->cpu_data()[index]);
//                     
// //                    printf("index = %d, label_value =  %d\n", index, label_value);
//                     bp_map_cpu[index] = static_cast<int>(bp_map_cpu2[index]);
// //                     if (bp_map_cpu[index] == 0 && prob_data_cpu_m) {
// //                         prob_data_cpu_m[index * dim + label_value] = (Dtype)1.;
// //                         for (int ind_cls = 0; ind_cls < inner_num2_; ind_cls++) {
// //                             if (ind_cls != label_value) {
// //                                 prob_data_cpu_m[index * dim + ind_cls] = (Dtype)0.0;
// //                             }
// //                         }
// //                     }
//                     
// #if 0
//                     if (bp_map_cpu[index] == 0 && prob_data_cpu_m) {
//                         memset((prob_data_cpu_m + index * dim), 0, sizeof(Dtype) * dim);
//                         prob_data_cpu_m[index * dim + label_value] = (Dtype)1.;
//                     }
// #endif
//                 }
//                 LOG(INFO) << "bp_map_cpu to prob_data_cpu_m finished";
            }
            
            // generate new bp to be used in the next step
            if (bp_size_ > 0) {
                if ( (bottom.size()>=3) && (nms_threshold_ > 0)  )
                {
                    const Dtype* bottom_rois = bottom[bottom.size()-1]->cpu_data();
                    const Dtype* labels =  bottom[1]->cpu_data();
//                     printf("before NMS\n");
                    nms<Dtype>(bottom_rois, prob_data_cpu, labels, bp_map_cpu, nms_threshold_, nthreads, dim);
//                     printf("after NMS\n");
                }
//                LOG(INFO)<<"total_gpu_num = "<<total_gpu_num<<", Samples_in_loop = "<<Samples_in_loop;
                // stat the real number of positive and negative samples in val samples per gpu.
                for (int gpu_index = 0; gpu_index < total_gpu_num; gpu_index++) {
                    init_val_index_per_gpu = gpu_index * total_rois_per_gpu;
                    
                    int real_positive_num = 0, real_negative_num = 0;
                    for (int index_inside_per_val = 0; index_inside_per_val < Samples_in_loop; index_inside_per_val++) {//OY
                        int current_val_index = init_val_index_per_gpu + index_inside_per_val;
                        const int label_value = static_cast<int>(bottom[1]->cpu_data()[current_val_index]);
//                         LOG(INFO) << "label_value: " << label_value;
                        
//                         if (current_val_index ==0)
//                             LOG(INFO)<<"bp_size_: " << bp_size_ <<"current_val_index = "<<current_val_index<<", label_value = "<<label_value << " bp_map_cpu: " << bp_map_cpu[current_val_index];
//                        printf("current_val_index = %d, label_value =  %d, bp_map_cpu: %d\n", current_val_index, label_value, bp_map_cpu[current_val_index]);
//                         LOG(INFO)<<"current_val_index = "<<current_val_index<<", label_value = "<<label_value << " bp_map_cpu: " << bp_map_cpu[current_val_index];
//                          if (bp_map_cpu[current_val_index] == 0)
//                             continue;
                       if (label_value != 0) {
                            real_positive_num++;
                        } else {
                            real_negative_num++;
                        }
                    }
//                     LOG(INFO)<<"bp_size_: " << bp_size_ <<" 1 real_positive_num = "<<real_positive_num<<", real_negative_num = "<<real_negative_num;
                    p_n_sample_num_per_gpu[gpu_index][0] = real_positive_num;
                    p_n_sample_num_per_gpu[gpu_index][1] = real_negative_num;
                    //LOG(INFO)<<"p_n_sample_num_per_gpu[gpu_index][0] = "<<p_n_sample_num_per_gpu[gpu_index][0];
                    //LOG(INFO)<<"p_n_sample_num_per_gpu[gpu_index][1] = "<<p_n_sample_num_per_gpu[gpu_index][1];
                }
//             LOG(INFO) << "real_positive_num  real_negative_num finished";
                
                //LOG(INFO)<<"before hard";
                // According to current or previous classification score,
                // choose Sample_num_positive_per_gpu  and Sample_num_negative_per_gpu samples
                for (int gpu_index = 0; gpu_index < total_gpu_num; gpu_index++) {
                    init_val_index_per_gpu = gpu_index * total_rois_per_gpu;
                    std::vector<std::pair<Dtype, int> > fw_positive_map;
                    std::vector<std::pair<Dtype, int> > fw_negative_map;
                    std::pair<Dtype, int> fw_sample;
                    int real_positive_num = p_n_sample_num_per_gpu[gpu_index][0];
                    int real_negative_num = p_n_sample_num_per_gpu[gpu_index][1];
//                    LOG(INFO)<<"gpu_index = "<<gpu_index;
//                    printf("302: real_positive_num = %d, real_negative_num =%d\n", real_positive_num, real_negative_num);
                    for (int index_inside_per_val = 0; index_inside_per_val < Samples_in_loop; index_inside_per_val++) {  //OY
                        int current_val_index = init_val_index_per_gpu + index_inside_per_val;
                        const int label_value = static_cast<int>(bottom[1]->cpu_data()[current_val_index]);
//                         if (bp_map_cpu[current_val_index] == 0)
//                             continue;
                        if (label_value >0) { //OY
                            // sample positive
//                             printf("%d, prob_data %d, label %d:  %.5f\n",  Caffe::MPI_my_rank(), current_val_index, label_value, prob_data_cpu[current_val_index*dim + label_value]);
//                             LOG(INFO) << Caffe::MPI_my_rank() << " prob_data " << current_val_index << " " << prob_data_cpu[current_val_index*dim + label_value] << " label: " << label_value;
                            fw_sample = std::make_pair(prob_data_cpu[current_val_index*dim + label_value], current_val_index);
                            fw_positive_map.push_back(fw_sample);
                        }
                        if (label_value < 1) { //OY
                            // mine the hardest negative samples
//                             printf("%d, prob_data %d, label %d:  %.5f\n",  Caffe::MPI_my_rank(), current_val_index, label_value, prob_data_cpu[current_val_index*dim + label_value]);
//                             LOG(INFO) << Caffe::MPI_my_rank()  << "prob_data " << current_val_index << " " << prob_data_cpu[current_val_index*dim + label_value] << "label: " << label_value;
                            fw_sample = std::make_pair(prob_data_cpu[current_val_index*dim + label_value], current_val_index);
                            fw_negative_map.push_back(fw_sample);
                        }
                    }
//                     LOG(INFO)<<"before sort";
                    //randomly sample positive samples
                     //Choose top postives
                    if (!rand_pos_)
                        std::sort(fw_positive_map.begin(), fw_positive_map.end());
                    std::sort(fw_negative_map.begin(), fw_negative_map.end());
//                         LOG(INFO)<<"after sort";
//                         LOG(INFO)<<"fw_positive_map.size() = "<<fw_positive_map.size()<<", fw_negative_map.size() = "<<fw_negative_map.size();
//                         LOG(INFO)<<"Samples_in_loop = "<<Samples_in_loop;
//                         LOG(INFO)<<"Sample_num_positive_per_gpu = "<<Sample_num_positive_per_gpu<<", real_positive_num = "<<real_positive_num;
//                         LOG(INFO)<<"Sample_num_negative_per_gpu = "<<Sample_num_negative_per_gpu <<", real_negative_num = "<<real_negative_num;
                        int PosNum;
                        PosNum = 0;
                        int NegNum;
                        NegNum = 0;
                        if (cascade_type_==0) { // cascade by sorting samples
//                             const Dtype* bp_map_cpu2 = bottom[2]->cpu_data();
                            int Rand_pos_num = floor(Sample_num_positive_per_gpu * rand_ratio_);
                            int Step;
                            int ind;
                            ind = 0;
                            for (; ind < real_positive_num; ind++) { //OY
                                //Choose top postives
//                             if ( PosNum < Sample_num_positive_per_gpu )
                                // take the previous bp map into consideration
                                if ( (PosNum < Sample_num_positive_per_gpu-Rand_pos_num) && bp_map_cpu[fw_positive_map[ind].second])
                                {
                                    bp_map_cpu[fw_positive_map[ind].second] = 1;
                                    PosNum ++;
                                } 
                                else
                                {
                                    if (bp_map_cpu[fw_positive_map[ind].second])
                                        bp_map_cpu[fw_positive_map[ind].second] = 2;
                                    else
                                        bp_map_cpu[fw_positive_map[ind].second] = 0;                                        
                                }
/*                                else {
                                    bp_map_cpu[fw_positive_map[ind].second] = 0;
                                }*/
                            }
                            if ((rand_ratio_ > 0)  || (PosNum<Sample_num_positive_per_gpu ) )
                            {
                                int idx = 0;
                                if (Rand_pos_num > 1)
                                    Step = floor((real_positive_num -1)/Rand_pos_num);
                                else 
                                    Step = 1;
                                if (Step < 1)
                                    Step = 1;
                                while (Step >=1)
                                {
                                    ind = idx;
                                    for (; ind < real_positive_num; ind+=Step) { //OY
                                        //Choose random postives  but take the previous bp map into consideration
                                        //do not take the previous bp map into consideration
                                        if ( PosNum < Sample_num_positive_per_gpu&&  (bp_map_cpu[fw_positive_map[ind].second]>1) )
                                        {
                                            if ( bp_map_cpu[fw_positive_map[ind].second] == 2 )
                                            {
                                                bp_map_cpu[fw_positive_map[ind].second] = 1;
                                                PosNum ++;
                                            }
                                        }
                                    }
                                    Step--;
                                    if ( PosNum == Sample_num_positive_per_gpu )
                                        break;                                    
                                }
                            }
                            int PosNum3 = 0;
                            for (ind=0; ind < real_positive_num; ind++) { //OY
                                if (bp_map_cpu[fw_positive_map[ind].second]==2)
                                {
                                    bp_map_cpu[fw_positive_map[ind].second] = 0;
                                }
                                if (bp_map_cpu[fw_positive_map[ind].second]==1)
                                    PosNum3++;                                
                            }
                            if (PosNum3 != PosNum)
                                printf("Warning 406: PosNum: %d, PosNum2: %d\n", PosNum, PosNum3);

                            int extra_neg = 0;
                            if (Sample_num_positive_per_gpu > PosNum)
                            {
                                extra_neg = Sample_num_positive_per_gpu - PosNum;
//                                  printf("407: add  %d negative samples\n ", extra_neg);
                            }
                            int Ran_neg_num = floor((Sample_num_negative_per_gpu+extra_neg) * rand_ratio_);
                            ind = real_positive_num;
                            for (; ind < real_positive_num+real_negative_num; ind++) {//OY
                                // take the previous bp map into consideration
                                if (NegNum < Sample_num_negative_per_gpu+extra_neg-Ran_neg_num && bp_map_cpu[fw_negative_map[ind - real_positive_num].second])
//                             Choose top negatives
//                             if (NegNum < Sample_num_negative_per_gpu )
                                {
                                    bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 1;
                                    NegNum++;
                                }
                                else
                                {
                                    if (bp_map_cpu[fw_negative_map[ind - real_positive_num].second])
                                        bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 2;
                                    else
                                        bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 0;                                        
                                }
/*                                else {
                                    bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 0;
                                }*/
                            }
//                             printf("421: %d negative samples included at the first step\n ", NegNum);
                            if ( (rand_ratio_ > 0) || (PosNum+NegNum)<Sample_num_per_gpu )
                            {
//                                 printf("398: %d negative samples included \n ", NegNum);
                                if (Ran_neg_num > 0)
                                    Step = floor((real_negative_num -real_positive_num-1)/Ran_neg_num);
                                else 
                                    Step = 1;
                                if (Step < 1)
                                    Step = 1;
                                while (Step >=1)
                                {
                                    ind = real_positive_num;
                                    for (; ind < real_positive_num+real_negative_num; ind+=Step) {//OY
                                        if ((NegNum < Sample_num_negative_per_gpu+extra_neg) && (bp_map_cpu[fw_negative_map[ind - real_positive_num].second]>1) )
//                             Choose random negatives but take the previous bp map into consideration
                                        {
                                            if ( bp_map_cpu[fw_negative_map[ind - real_positive_num].second] == 2 )
                                            {
                                                bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 1;      
                                                NegNum++;                                        
                                            }
                                        }
                                    }
                                    Step--;
                                    if (NegNum == Sample_num_negative_per_gpu+extra_neg)
                                        break;
                                }
//                                  printf("451: %d negative samples included at the second step\n ", NegNum);
                                
                            }
                            if ( (PosNum+NegNum)<Sample_num_per_gpu )
                            {
                                Step3_Cnt++;                                                                
                                if (Step3_Cnt%100==0)
                                    printf("597: %d pos %d neg samples ->  ", PosNum, NegNum);
                                Step = 1;
                                ind = real_positive_num;
                                for (; ind < real_positive_num+real_negative_num; ind+=Step) {//OY
                                    // do not take the previous bp map into consideration
                                    if ((NegNum < Sample_num_negative_per_gpu+extra_neg) && !bp_map_cpu[fw_negative_map[ind - real_positive_num].second] )
                                    {
                                           bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 1;
                                           NegNum++;
                                    }
                                }
                                if ( (PosNum+NegNum)<Sample_num_per_gpu )
                                {
                                    ind = 0;
                                    Step = 1;
                                    printf("615: %d pos %d neg samples ->  ", PosNum, NegNum);
                                    for (; ind < real_positive_num+real_negative_num; ind+=Step) 
                                    {//OY
                                        // do not take the previous bp map into consideration
                                        if ( ((PosNum+NegNum)<Sample_num_per_gpu) && !bp_map_cpu[fw_positive_map[ind].second] )
                                        {
                                            bp_map_cpu[fw_positive_map[ind].second] = 1;
                                            PosNum++;
                                        }
                                    }
                                    printf("%d pos %d neg samples in the third step, Step3_Cnt: %d\n ", PosNum, NegNum);                                    
                                }
                                if (Step3_Cnt%100==0)
                                    printf("%d neg samples in the third step, Step3_Cnt: %d\n ", NegNum, Step3_Cnt);
                           }
                            int NegNum2 = 0;
                            for (ind = real_positive_num; ind < real_positive_num+real_negative_num; ind++) {//OY
                                if (bp_map_cpu[fw_negative_map[ind - real_positive_num].second]==2)
                                    bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 0;
                                if (bp_map_cpu[fw_negative_map[ind - real_positive_num].second]==1)
                                    NegNum2++;
                            }
                            if (NegNum2 != NegNum)
                                printf("Warning 500: NegNum: %d, NegNum2: %d\n", NegNum, NegNum2);
                            if ( ((PosNum+NegNum)<Sample_num_per_gpu) || ((PosNum+NegNum)>Sample_num_per_gpu) )
                            {
                                printf("410: Sample_num_per_gpu = %d, Real bp size =%d+%d=%d, real_positive_num: %d, real_negative_num: %d \n", Sample_num_per_gpu, PosNum, NegNum, PosNum+NegNum, real_positive_num, real_negative_num);
                                LOG(INFO)<<"Warning!!! Ran_neg_num: " << Ran_neg_num << " Rand_pos_num: " << Rand_pos_num ;                               
                                LOG(INFO)<<"Warning!!! bp_map_cpu regenerated " << " Sample_num_per_gpu: " << Sample_num_per_gpu << " Real bp size: " << PosNum << " + " << NegNum << "=" << PosNum+NegNum << ", real_positive_num: " << real_positive_num << ", real_negative_num: " << real_negative_num; 
                            }

                        }
                        else {
                            for (int ind = 0; ind < real_positive_num; ind++) { //OY
                                //Choose postives with score > threshold
                                // take the previous bp map into consideration
//                                 LOG(INFO) << "pos data: " << fw_positive_map[ind].first << " Threshold:" << threshold_;
                                if ( bp_map_cpu[fw_positive_map[ind].second])
                                {
                                    if ( fw_positive_map[ind].first < threshold_  ) 
                                    {
                                        bp_map_cpu[fw_positive_map[ind].second] = 1;
                                        PosNum ++;
                                    }
                                    else
                                        bp_map_cpu[fw_positive_map[ind].second] = 2;
                                } else {
                                    bp_map_cpu[fw_positive_map[ind].second] = 0;
                                }
                            }
                            int Pos_Keep = Sample_num_positive_per_gpu*keep_ratio_;
                            if (Pos_Keep<1) 
                                Pos_Keep = 1;
                            for (int ind = 0; ind < real_positive_num; ind++) { //OY
                                if ( bp_map_cpu[fw_positive_map[ind].second] > 1)
                                {
                                    if ( PosNum<Pos_Keep )
                                    {
                                        bp_map_cpu[fw_positive_map[ind].second] = 1;
                                        PosNum++;
                                    }
                                    else
                                        bp_map_cpu[fw_positive_map[ind].second] = 0;
                               }
                            }
//                             int N_neg2 = 0;
                            for (int ind = real_positive_num; ind < real_positive_num+real_negative_num; ind++) {//OY
                                // take the previous bp map into consideration
//                                 LOG(INFO) << "neg data: " << fw_negative_map[ind - real_positive_num].first << " Threshold:" << threshold_;
                                if (bp_map_cpu[fw_negative_map[ind - real_positive_num].second])
                                // Choose negatives with score > threshold
                                {
                                    if ( (fw_negative_map[ind - real_positive_num].first < threshold_) )
                                    {
                                        bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 1;
                                        NegNum++;
                                    }
                                    else
                                    {
                                        bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 2;
//                                         N_neg2++;
                                    }
                                } else {
                                    bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 0;
                                }
                            }
                            int Neg_Keep = Sample_num_negative_per_gpu*keep_ratio_;
                            if (Neg_Keep<1)
                                Neg_Keep = 1;
                            for (int ind = real_positive_num; ind < real_positive_num+real_negative_num; ind++) {//OY
                                if ( bp_map_cpu[fw_negative_map[ind - real_positive_num].second] > 1)
                                {
                                    if ( NegNum<Neg_Keep )
                                    {
                                        bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 1;
                                        NegNum++;
                                    }
                                    else
                                        bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 0;
                               }
                            }
                            if ( PosNum<Pos_Keep )
                            {
                                Ratio_keep_Cnt++;
                                if (Ratio_keep_Cnt%100==0)
                                    printf("699: %d + samples -> ", PosNum);
                                int Step = 1;
                                int ind = 0;
                                for (; ind < real_positive_num; ind+=Step) {//OY
                                    // do not take the previous bp map into consideration
                                    if ((PosNum<Pos_Keep) && !bp_map_cpu[fw_positive_map[ind].second] )
//                             Choose random negatives
                                    {
                                        bp_map_cpu[fw_positive_map[ind].second] = 1;
                                        PosNum++;
                                    }
                                }
                                
                                if (Ratio_keep_Cnt%100==0)
                                    printf("%d pos samples by Pos_Keep, Ratio_keep_Cnt: %d\n ", PosNum, Ratio_keep_Cnt);
                            }
                            if ( (NegNum)<Neg_Keep )
                            {
                                Ratio_keep_Cnt++;
                                if (Ratio_keep_Cnt%100==0)
                                    printf("715: %d  -> ", NegNum);
                                int Step = 1;
                                int ind = real_positive_num;
                                for (; ind < real_positive_num+real_negative_num; ind+=Step) {//OY
                                    // do not take the previous bp map into consideration
                                    if ((NegNum < Neg_Keep) && !bp_map_cpu[fw_negative_map[ind - real_positive_num].second] )
//                             Choose random negatives
                                    {
                                        bp_map_cpu[fw_negative_map[ind - real_positive_num].second] = 1;
                                        NegNum++;
                                    }
                                }
                                if (Ratio_keep_Cnt%100==0)
                                    printf(" %d neg samples by Neg_Keep, Ratio_keep_Cnt: %d\n ", NegNum, Ratio_keep_Cnt);
                            }
                            
                        }
                }
                
                
                
                // Get some statistics about the chosen samples
                Dtype p_score, n_score;
                p_score=0; n_score=0; negative = 0; positive = 0;
//                 printf("before stats in softmax_cascade_loss_layer.cu");
                for (int index = 0; index < nthreads; index++) {
//                     if (Caffe::MPI_my_rank() == 0) {
//                         LOG(INFO)<< "outer_num_: " << outer_num_ <<" inner_num_:" << inner_num_ << " nthreads: " << nthreads <<"index = "<<index;
//                     }
                    if (bp_map_cpu[index] == 1) {
                        const int label_value = static_cast<int>(bottom[1]->cpu_data()[index]);
//                         if (Caffe::MPI_my_rank() == 0) {
//                             LOG(INFO)<<", label = "<<label_value;
//                             LOG(INFO)<<"prob: "<<prob_data_cpu[index*dim + label_value];
//                         }
                        if (label_value == 0) {
                            negative++;
                            n_score+=prob_data_cpu[index*dim + label_value];
                        } else {
                            positive++;
                            p_score+= prob_data_cpu[index*dim + label_value];
                        }
                    }
                }
//                 printf("after stats 1 in softmax_cascade_loss_layer.cu\n");
//                 if (positive<0)
//                     positive = 1;
//                 if (negative<0)
//                     negative = 1;
                Pos_scores_ += p_score/(0.001+positive);
                Neg_scores_ += n_score/(0.001+negative);
                PosCnt_+=positive;
                NegCnt_+=negative;
                BPCnt_+= positive+negative;
                if ( ((iter_cnt_%200==0)||(iter_cnt_<5)) && (Caffe::MPI_my_rank() == 0) )
                    LOG(INFO)<<"fw_size: "<<nthreads <<" bp_size_: " << bp_size_ << "bp in:" << bp_cnt_in  << " real size: " << PosCnt_/iter_cnt_ << " + " << NegCnt_/iter_cnt_ << "=" << BPCnt_/iter_cnt_ << " score: +: " << Pos_scores_/iter_cnt_ << " -:"<<Neg_scores_/iter_cnt_;
//                 printf("after stats 2 in softmax_cascade_loss_layer.cu\n");

//                     LOG(INFO)<<"fw_size: "<<nthreads <<" bp_size_: " << bp_size_ << "bp in:" << bp_cnt_in << " out: " << positive << " + " << negative << "=" << BPCnt_/iter_cnt_ << " score: +: " << Pos_scores_/iter_cnt_ << " -:"<<Neg_scores_/iter_cnt_;
//                     LOG(INFO)<<"fw_size: "<<nthreads <<" bp_size_: " << bp_size_ << "bp in:" << bp_cnt_in << " out: " << PosCnt_/iter_cnt_ << " + " << NegCnt_/iter_cnt_ << "=" << BPCnt_/iter_cnt_ << " score: +: " << Pos_scores_/iter_cnt_ << " -:"<<Neg_scores_/iter_cnt_;
           }
            //LOG(INFO)<<"here";
            bp_map->mutable_gpu_data();

        }
                
         //LOG(INFO) << "bp_map_cpu2";
       const Dtype* bp_map_gpu;
       if (hard_mining_)
       {
           if (nms_threshold_ > 0)
               bp_map_gpu = bottom[bottom.size()-2]->gpu_data();
           else
               bp_map_gpu = bottom[bottom.size()-1]->gpu_data();
       }
        else
            bp_map_gpu = 0;
        Dtype* prob_data = prob_.mutable_gpu_data();

        // zx, cascade, add "hard_mining_" argument.
        // NOLINT_NEXT_LINE(whitespace/operators)
        //LOG(INFO) << "SoftmaxLossForwardGPU2";
        SoftmaxLossForwardGPU2<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
                CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, prob_map->gpu_data(), label, NULL, loss_data,
                outer_num_, dim, inner_num_, has_ignore_label_, hard_mining_, ignore_label_, counts, bp_map_gpu);
        
        Dtype loss = Dtype(0.0);
        caffe_gpu_asum(nthreads, loss_data, &loss);
        if (Caffe::MPI_my_rank() == 0) {
            //LOG(INFO)<<"cls_loss = "<<loss;
        }
//         LOG(INFO) << "normalize";
        if (normalize_) {
            Dtype count;
            caffe_gpu_asum(nthreads, counts, &count);
            if (hard_mining_ ) {
//                 LOG(INFO)<<"loss: fw_size: "<<nthreads<<", loss_count: "<<count<<", bp_size: "<<(positive+negative)<<", positive: "<<positive/(Dtype)(positive+negative) * 100<<"%, "<<"negative: "<<negative/(Dtype)(positive+negative) * 100<<"%.";
            }
            //else if (Caffe::MPI_my_rank() == 0){
            //  LOG(INFO)<<"loss1: fw_size: "<<nthreads<<", loss_count: "<<count<<", p+n = "<<(positive+negative)<<", pos_ratio: "<<positive/(Dtype)(positive+negative) * 100<<"%"<<", neg_ratio: "<<negative/(Dtype)(positive+negative) * 100<<"%";
            //}
            bp_count_ = count;
            if (bp_count_<1)
            {
                LOG(INFO) << "Warning bp_count_ is 0!!";
                bp_count_ = 1;
            }
            if (count != Dtype(0.0))
                loss /= count;
        } else {
            bp_count_ = outer_num_;
            loss /= outer_num_;
//              if (Caffe::MPI_my_rank() == 0) {
//                 LOG(INFO)<<"loss: fw_size: "<<nthreads<<", loss_count: "<<bp_count_<<", bp_size: "<<(positive+negative)<<", positive: "<<positive/(Dtype)(positive+negative) * 100<<"%, "<<"negative: "<<negative/(Dtype)(positive+negative) * 100<<"%.";
//             }
       }
        top[0]->mutable_cpu_data()[0] = loss;
        
        // zx, hard, begin
        // zx, original implementation
        //if (top.size() == 2) {
        //  top[1]->ShareData(prob_);
        //}
        
        //LOG(INFO) << "top_1";
        if (bp_size_>0 ) {
//         if (bp_size_>0 && Caffe::MPI_my_rank() == 0) {
            Dtype *top_1 = top[1]->mutable_cpu_data();
            const int *bp_map_cpu = bp_map->cpu_data();
            for (int index = 0; index < nthreads; index++) {
                top_1[index] = static_cast<Dtype>(bp_map_cpu[index]);
//                 printf("%d, bp_map out %d:  %d\n",  Caffe::MPI_my_rank(), index, bp_map_cpu[index]);
//                 LOG(INFO) << Caffe::MPI_my_rank() << "bp_map out " << index << ": " << bp_map_cpu[index];
            }
        }
        // zx, hard, end
    }
    
    template <typename Dtype>
            __global__ void SoftmaxLossBackwardGPU2(const int nthreads, const Dtype* top,
            const Dtype* label, const Dtype *weight, Dtype* bottom_diff, const int num, const int dim,
            const int spatial_dim, const bool has_ignore_label_,
            const int ignore_label_, Dtype* counts) {
        // zx, dim == 201, spatial_dim == 1, nthreads = 48
        const int channels = dim / spatial_dim;
        
        CUDA_KERNEL_LOOP(index, nthreads) {
            const int n = index / spatial_dim;
            const int s = index % spatial_dim;
            const int label_value = static_cast<int>(label[n * spatial_dim + s]);
            
            // zx, has_ignore_label_ == false
            if (has_ignore_label_ && label_value == ignore_label_) {
                for (int c = 0; c < channels; ++c) {
                    bottom_diff[n * dim + c * spatial_dim + s] = 0;
                }
                counts[index] = 0;
            } else {
                // zx, weight == NULL
                bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
                
                if (weight != NULL){
                    const Dtype weight_value = static_cast<Dtype>(weight[n * spatial_dim + s]);
                    for (int k = 0; k < channels; ++k)
                        bottom_diff[n * dim + k * spatial_dim + s] *= weight_value;
                    
                    counts[index] = weight_value;
                }
                else
                    counts[index] = 1;
            }
        }
    }
    
    template <typename Dtype>
            void SoftmaxWithCascadeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        if (propagate_down[1]) {
            LOG(FATAL) << this->type()
            << " Layer cannot backpropagate to label inputs.";
        }
        if (propagate_down[0]) {
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const Dtype* prob_data = prob_.gpu_data();
            const Dtype* top_data = top[0]->gpu_data();
            caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
            const Dtype* label = bottom[1]->gpu_data();
            // zx, bottom.size() == 2, has_weight == false
            // zx, cascade, begin
            // original implementation, begin
            //bool has_weight = bottom.size() >= 3;
            // original implementation, end
            // zx, cascade, end
//             bool has_weight =( (bottom.size() == 3)&&(nms_threshold_==0) || (bottom.size() == 4) );
            const Dtype* weight = NULL;
            const Dtype* bp_map_gpu;
            if (hard_mining_)
            {
                if (nms_threshold_ > 0)
                    bp_map_gpu = bottom[bottom.size()-2]->gpu_data();
                else
                    bp_map_gpu = bottom[bottom.size()-1]->gpu_data();
            }
            else
                bp_map_gpu = NULL;
//             if (has_weight)
//                 weight = bottom[2]->gpu_data();
            // zx, outer_num_ == 48, inner_num_ == 1, prob_.count() = 48*201 = 9648, dim = 9648/48=201
            const int dim = prob_.count() / outer_num_;
            const int nthreads = outer_num_ * inner_num_;
            // Since this memory is never used for anything else,
            // we use to to avoid allocating new GPU memory.
            Dtype* counts = prob_.mutable_gpu_diff();
            
            // zx, weight == NULL,
            // NOLINT_NEXT_LINE(whitespace/operators)
            SoftmaxLossBackwardGPU2<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
                    CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bp_map_gpu, bottom_diff,
                    outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
            
            // zx, cascade, begin, change "const Dtype loss_weight = top[0]->cpu_diff()[0];" to þýœloss_weight = 1;þýþý
            const Dtype loss_weight = top[0]->cpu_diff()[0];
            //const Dtype loss_weight = 1;
            // zx, cascade, end
//             LOG(INFO) << "loss_weight: " << loss_weight;
            if (normalize_) {
//                 Dtype count;
//                 caffe_gpu_asum(nthreads, counts, &count);
                caffe_gpu_scal(prob_.count(), bp_count_ == Dtype(0.0) ? Dtype(0.0) : (loss_weight / bp_count_), bottom_diff);
            } else {
                caffe_gpu_scal(prob_.count(), loss_weight / bp_count_, bottom_diff);
            }
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithCascadeLossLayer);
    
}  // namespace caffe
