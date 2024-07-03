#include <set>
#include <cstdint>
#include <cstddef>

/*! \brief Bounding box data structure with coordinates and confidence score. */
struct bbox_t {
    float x1, y1, x2, y2;
    float score;
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
            float intersection_area = std::max(0.0f, x2 - x1 + 1) * std::max(0.0f, y2 - y1 + 1);

            // calculate the area of the union of the two rectangles
            float curr_box_area = (curr_box.x2 - curr_box.x1 + 1) * (curr_box.y2 - curr_box.y1 + 1);
            float prev_box_area = (prev_box.x2 - prev_box.x1 + 1) * (prev_box.y2 - prev_box.y1 + 1);
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
        static void nms(struct bbox_t *boxes, size_t num_boxes, float confidence_threshold, float iou_threshold, struct bbox_t *keep, size_t *num_keep) {

            // initialize the num_keep variable
            *num_keep = 0;
            
            // sort the boxes by descending score
            std::multiset<bbox_t,bbox_comparator> bboxes;
            for (int i = 0; i < num_boxes; i++) {
                bboxes.insert(boxes[i]);
            }

            // keep a set of the boxes to keep
            std::set<bbox_t,bbox_comparator> keep_set;
            
            // iterate through the sorted boxes
            for (const auto& it : bboxes) {
                
                // check if the score of the current box is below the confidence threshold
                if (it.score < confidence_threshold) {
                    break;
                }

                // check if the current box overlaps with any of the boxes to keep
                bool keep_box = true;
                for(const auto& keep_it : keep_set) {
                    if (calculate_iou(it, keep_it) > iou_threshold) {
                        keep_box = false;
                        break;
                    }
                }
                // if the box does not overlap with any of the boxes to keep, add it to the set of boxes to keep
                if (keep_box) {
                    keep_set.insert(it);
                    *num_keep += 1;
                }
            }
            
        }

};

/*! \brief For integration usage. */
void nms(struct bbox_t *boxes, size_t num_boxes, float confidence_threshold, float iou_threshold, struct bbox_t *keep, size_t *num_keep) {
    NMS::nms(boxes, num_boxes, confidence_threshold, iou_threshold, keep, num_keep);
}
