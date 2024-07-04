#include <iostream>
#include <set>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

/*! \brief Bounding box data structure with coordinates and confidence score. */
struct bbox_t {
    float x1, y1, x2, y2;
    float score;
    std::uint8_t class_id;
};

/*! \brief Descending order by score bounding box comparator. */
struct bbox_comparator {
    // sort in descending order
    bool operator()(const bbox_t &a, const bbox_t &b) const {
        return a.score < b.score;
    }
};

/*! \brief Non-Maximum Suppression Intersection-Over-Union Algorithm implementation. */
class NMS {

    public:

        /*! \brief Calculate the Intersection-Over-Union (IoU) of two bounding boxes. 
        *
        *  \param curr_box Current bounding box.
        *  \param prev_box Previous bounding box.
        *  \return Intersection-Over-Union (IoU) of the two bounding boxes.
        */
        static float calculate_iou(struct bbox_t curr_box, struct bbox_t prev_box) {
            // get the coordinates of the intersection rectangle
            float x1 = std::max(curr_box.x1, prev_box.x1);
            float y1 = std::max(curr_box.y1, prev_box.y1);
            float x2 = std::min(curr_box.x2, prev_box.x2);
            float y2 = std::min(curr_box.y2, prev_box.y2);

            // calculate the area of the intersection rectangle
            float intersection_area = std::max(0.0f, x2 - x1 + 1.0f) * std::max(0.0f, y2 - y1 + 1.0f);

            // calculate the area of the union of the two rectangles
            float curr_box_area = (curr_box.x2 - curr_box.x1 + 1.0f) * (curr_box.y2 - curr_box.y1 + 1.0f);
            float prev_box_area = (prev_box.x2 - prev_box.x1 + 1.0f) * (prev_box.y2 - prev_box.y1 + 1.0f);
            float union_area = curr_box_area + prev_box_area - intersection_area;

            // calculate the intersection over union
            float iou = intersection_area / union_area;

            return iou;
        }

        /*! \brief Non-Maximum Suppression (NMS) algorithm implementation. 
            * 
            *  \param boxes Array of bounding boxes.
            *  \param num_boxes Number of bounding boxes.
            *  \param confidence_threshold Confidence threshold for bounding boxes.
            *  \param iou_threshold Intersection-Over-Union threshold for bounding boxes.
            *  \param keep Array of bounding boxes to keep.
            *  \param num_keep Number of bounding boxes kept.
        */
        static void nms(float *boxes, float *scores, std::uint32_t num_boxes, std::uint8_t num_classes, float confidence_threshold, float iou_threshold, struct bbox_t **keep, std::uint32_t *num_keep) {

            // initialize the num_keep variable
            *num_keep = 0;

            // allocate the memory for the set of boxes to keep
            *keep = (struct bbox_t *)malloc(num_boxes * sizeof(struct bbox_t));
            if(*keep == NULL) {
                return;
            }
            
            // sort the boxes by descending score
            std::multiset<bbox_t,bbox_comparator> bboxes;
            for (int i = 0; i < num_boxes; i++) {
                // find the class with the highest score
                float max_score = 0.0f;
                std::uint8_t max_class = 0;
                for (int j = 0; j < num_classes; j++) {
                    if (scores[i*num_classes+j] > max_score) {
                        max_score = scores[i*num_classes+j];
                        max_class = j;
                    }
                }
                bboxes.insert({boxes[i*4], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3], max_score, max_class});
            }

            // std::cout << "I'M HERE" << std::endl;
            
            // iterate through the sorted boxes
            int i = 0;
            for (const auto& it : bboxes) {
                
                // check if the score of the current box is below the confidence threshold
                if (it.score < confidence_threshold) {
                    continue;
                }

                // check if the current box overlaps with any of the boxes to keep
                bool keep_box = true;
                for(std::uint32_t i = 0; i < (*num_keep); i++) {
                    if (calculate_iou(it, (*keep)[i]) > iou_threshold) {
                        keep_box = false;
                        break;
                    }
                }

                // if the box does not overlap with any of the boxes to keep, add it to the set of boxes to keep
                if (keep_box) {
                    (*keep)[*num_keep] = it;
                    (*num_keep) += 1;
                }
            }
            
        }

};

/*! \brief For integration usage. */
#ifdef __cplusplus
extern "C" {
#endif
void nms(float *boxes, float *scores, std::uint32_t num_boxes, std::uint8_t num_classes, float confidence_threshold, float iou_threshold, struct bbox_t **keep, std::uint32_t *num_keep) {
    NMS::nms(boxes, scores, num_boxes, num_classes, confidence_threshold, iou_threshold, keep, num_keep);
}
#ifdef __cplusplus
}
#endif
