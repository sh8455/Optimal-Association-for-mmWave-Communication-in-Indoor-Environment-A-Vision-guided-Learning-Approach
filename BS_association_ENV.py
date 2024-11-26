import numpy as np
import CalCoor
import Coordinates
import matlab.engine
from collections import Counter

K_B = 1.38064852e-23 # 볼츠만 상수
T = 290 # 절대온도
B = 400e6 # 대역폭 (MHz)
Pn = 10 * np.log10(K_B*T*B)+30
Pn_W = 10 ** ((Pn-30)/10)

P_tx_dBm = 23 # 송신 전력 (23 dBm)
P_tx_W = 10**((P_tx_dBm-30)/10)

eng = matlab.engine.start_matlab()
eng.cd("C:/Users/User/Desktop/graduate/DRL")
eng.addpath("C:/Users/User/Desktop/graduate/DRL")

class BsAssociation:
    def __init__(self, BS_num, UE_num):
         # 타임 스텝 카운트
        self.time_step = 0
        
        # 환경
        self.BS_num = BS_num
        self.UE_num = UE_num
        self.BS_budget = 5
        self.UE_locations, self.BS_locations = Coordinates.change_coor_np(Coordinates.UE_coordinates, Coordinates.BS_Coordinates, self.time_step)
        self.distance = []
        self.selected_BS = []
        self.BS = {}
        
        # 각 BS별 변수
        for i in range(self.BS_num):
            self.BS[i] = {
                'max_RT' : [], # 각 BS의 사람별 높은 RT 결과
                'choice_maxDR' : [],  # 각 BS가 선택한 사람별 최대 DR
            }
            
    def load_raytracing_results(self, BS_positions, UE_positions):
        for i in range(self.BS_num):
            # BS 위치와 UE 위치 전달 (MATLAB 형식으로 변환)
            BS_position_matlab = matlab.double(BS_positions[i].tolist())
            UE_positions_matlab = matlab.double(UE_positions.T.tolist())  # 2D array로 변환
            
            # MATLAB의 raytrace_simulation 호출
            result = eng.raytrace_simulation(BS_position_matlab, UE_positions_matlab, nargout=1)
            
            idx = 0
            for i in range(self.BS_num):
                RT = []
                for j in range(self.UE_num):
                    path_loss = result[idx]['PathLoss']
                    propagation_distance = result[idx]['PropagationDistance']
                    RT.append(np.array([path_loss, propagation_distance]))
                self.BS[i]['max_RT'] = RT
            
    def reset(self):
        # 환경
        self.time_step = 0
        self.UE_locations, self.BS_locations = Coordinates.change_coor_np(Coordinates.UE_coordinates, Coordinates.BS_Coordinates, self.time_step)
            
        self.distance = []
    
        for i in range(self.BS_num):
            d_result = []
            DRresult = []
            for j in range(self.UE_num):
                d = self.calDistance(self.UE_locations[j], self.BS_locations[i])
                d_result.append(d)
                DR = self.calDR(self.BS[i]['max_RT'][j][0])/1e9
                DRresult.append(DR)
            self.distance.append(np.array(d_result))
            self.BS[i]['choice_maxDR'] = np.array(DRresult)
            
        self.selected_BS = [0,0,0,0,0,0,0,0,0,0]
        
        self.load_raytracing_results(self.BS_locations, self.UE_locations)
        
        return self._get_state()
    
    def calDistance(self, UE_loc, BS_loc):
        distance = np.linalg.norm(BS_loc - UE_loc)
        
        return distance
    
    def calDR(self, pathloss):
        # 수신 전력 계산
        P_rx_dBm = P_tx_dBm - pathloss
        P_rx_W = 10**((P_rx_dBm-30)/10)
        
        # SNR 계산
        SNR_linear = P_rx_W/Pn_W
        
        # Data Rate 계산
        DR = B * np.log2(1+SNR_linear)
        
        return DR
    
    # 보상 계산
    def calReward(self, selected_BS):
        selected_maxDR = 0
        
        for i in range(self.UE_num):
            DR = self.BS[selected_BS[i]]['choice_maxDR'][i]
            selected_maxDR += DR
            
        return selected_maxDR
            
    # 스텝
    def step(self, action):
        print(f"step: {self.time_step}")
        print(f"action: {action}")
        
        self.selected_BS = action
        
        cnt_BSmatching = Counter(self.selected_BS)
        
        self.UE_locations, self.BS_locations = Coordinates.change_coor_np(Coordinates.UE_coordinates, Coordinates.BS_Coordinates, self.time_step)
            
        # 각 BS별로 각 UE에게 제공 가능한 가장 높은 DR 구성요소 추출
        self.load_raytracing_results(self.BS_locations, self.UE_locations) 
        
        # UE별로 DR과 거리 계산
        for i in range(self.BS_num):
            d_result = []
            DRresult = []
            for j in range(self.UE_num):
                d = self.calDistance(self.UE_locations[j], self.BS_locations[i])
                d_result.append(d)
                DR = self.calDR(self.BS[i]['max_RT'][j][0])/1e9
                DRresult.append(DR)  
            self.distance.append(d_result)
            self.BS[i]['choice_maxDR'] = np.array(DRresult) 
            
        self.time_step += 1
        
        done = False
        
        # BS의 매칭 개수 제약
        for i in range(self.BS_num):
            if cnt_BSmatching[i] >= 5:
                done = True
        
        if self.time_step == 24:
            done = True
            
        reward = self.calReward(action)
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        obs = []
        sum_pdis = []
        
        distance_BS1 = self.distance[0]
        distance_BS2 = self.distance[1]
        distance_BS3 = self.distance[2]
        distance_BS4 = self.distance[3]
        
        dr_1 = self.BS[0]['choice_maxDR']
        dr_2 = self.BS[1]['choice_maxDR']
        dr_3 = self.BS[1]['choice_maxDR']
        dr_4 = self.BS[1]['choice_maxDR']
        
        selected_BS = self.selected_BS
        
        for i in range(self.BS_num):
            propagation_dis = []
            for j in range(self.UE_num):
                propagation_dis.append(self.BS[i]['choice_maxRT'][j][0])
            sum_pdis.append(np.array(propagation_dis))
            
        sum_pdis1 = sum_pdis[0]
        sum_pdis2 = sum_pdis[1]
        sum_pdis3 = sum_pdis[2]
        sum_pdis4 = sum_pdis[3]

        obs = np.concatenate(
            [distance_BS1, distance_BS2, distance_BS3, distance_BS4, dr_1, dr_2, dr_3, dr_4, selected_BS, sum_pdis1, sum_pdis2, sum_pdis3, sum_pdis4]
        )

        return np.array(obs)
