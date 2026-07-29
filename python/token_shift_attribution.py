#token shift attribution:


Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
Eigen::Vector3d a(0.5, 3, -0.4);
Eigen::Vector3d Aa = A * a;
Eigen::VectorXd b(5);
Eigen::VectorXd Bb = B * b;
Eigen::MatrixXd A(3, 2); #matrix(0, 3, 2)
Eigen::MatrixXd B = A.transpose();
Eigen::MatrixXd C = (B * A).inverse();
Eigen::Vector3d V(1, 2, 3);
Eigen::Vector3d W(0, 1, 2);
double vDoW = V.dot(W);
Eigen::Vector3d vCrossW = V.cross(W);
Eigen::MatrixXd A = Eigen::MatrixXd::Random(7, 9);

Eigen::MatrixXd A = Eigen::MatrixXd::Random(7, 9);
double vDoW = V.dot(W);
Eigen::Vector3d W(0, 1, 2);

#the first queue is coming from the in_stk
#the second queue is coming from the out_stk



class MyQueue:
    def __init__(self):
        self.in_stk = []
        self.out_stk = []
    def push(self, X):
        self.in_stk.append(X)
    def pop(self):
        self.peek()
        return self.out_stk.pop()
    def peek(self):
        if not self.in_stk:
            while self.in_stk:
                self.out_stk.append(self.in_stk.pop())
        return self.out_stk[-1]
    def empty(self):
        return not self.in_stk and not self.out_stk




class MyQueue:
    def __init__(self):
        self.in_stk = []
        self.out_stk = []
    def push(self, X):
        self.in_stk.append(X)
    def pop(self):
        self.peek()
        return self.out_stk.pop()
    def peek(self):
        if not self.in_stk:
            while self.in_stk:
                self.out_stk.append(self.in_stk.pop())
        return self.out_stk[-1]
    def empty(self):
        return not self.in_stk and not self.out_stk


























































{
  "log_id": "20260727_140523_001",
  "timestamp": "2026-07-27T14:05:23.123+08:00",
  "window_id": "window_5min_1405",  // 5分钟滚动窗口标识
  "context": {
  "exp_group": "treatment_v2.1",
  "ad_slots": "feed_third",
  "user_cohort": "age_25_35_high_intent",
  "creative_type": "short_drama",
  "device_os": "iOS"
  },
  "model_snapshot":{
  "pred_bid_coeff": 1.32,
  "feature_hash": "ab3d8f9c",
  "embedding_cluster": 7
  },
  "label_realtime":{
  "is_show": 1,
  "is_click": 0,
  "click_cost": 0.05,
  "gmv_30min": null #d
  }#back to the json logs here.
}





















































