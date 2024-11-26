function RT_result = raytrace_simulation(BS_position, UE_positions)
    fc = 64e9; % 64 GHz
    lambda = physconst("lightspeed") / fc;

    % 안테나 배열 설정
    txArray = phased.URA('Size', [4, 4], 'ElementSpacing', lambda / 2);
    rxArray = phased.URA('Size', [4, 4], 'ElementSpacing', lambda / 2);

    % 송신기 배열 생성 (BS_position의 각 좌표에 대해 송신기 생성)
    numBS = size(BS_position, 2); % 송신기 수 (4개)
    numUE = size(UE_positions, 2); % 수신기 수 (10개)

    % 송신기와 수신기 각각에 대해 레이 트레이싱 수행
    idx = 1; 
    for i = 1:numBS
        % 송신기 설정 (BS_position의 각 항목에 대해 송신기 생성)
        tx = txsite('cartesian', ...
            'Antenna', txArray, ...
            'AntennaPosition', BS_position(:, i), ...
            'TransmitterFrequency', fc);
        
        for j = 1:numUE
            % 수신기 설정 (UE_positions의 각 항목에 대해 수신기 생성)
            rx = rxsite('cartesian', ...
                'Antenna', rxArray, ...
                'AntennaPosition', UE_positions(:, j));
            
            pm = propagationModel('raytracing', ...
                'Method', 'sbr', ...
                'MaxNumReflections', 2, ...
                'SurfaceMaterial', 'concrete');
            
            rays = raytrace(tx, rx, pm);
            
            % 각 tx, rx에 대해 가장 낮은 PathLoss를 가진 경로를 선택
            if isempty(rays)
                RT_result{idx} = struct('PathLoss', NaN, 'PropagationDistance', NaN);
            else
                [minPathLoss, index] = min([rays.PathLoss]); % 가장 작은 PathLoss 선택
                RT_result{idx} = struct('PathLoss', minPathLoss, 'PropagationDistance', rays(index).PropagationDistance);

            idx = idx + 1;  % 인덱스 증가
            end
        end
    end
end
