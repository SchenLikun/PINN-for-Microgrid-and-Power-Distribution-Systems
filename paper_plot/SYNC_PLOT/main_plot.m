clear;
clc;


%% 定义全局时间区间变量
% 这里定义的时间区间将用于后续的特定区间绘图
globalTimeStart = 0;  % 开始时间
globalTimeEnd = 5e5;    % 结束时间

%% 绘图文件
% List of file paths
filePaths = {
    'SYNC_variables_0318_TransAll.dat', 
    'SYNC_variables_0318_TransIn.dat', 
    'SYNC_variables_0318_TransNone.dat', 
    'SYNC_variables_0318_TransOut.dat'
};

% Preallocate cell array to hold the data from all files
allData = cell(1, length(filePaths));

% Load the data using your existing function
for i = 1:length(filePaths)
    allData{i} = load_and_process_data(filePaths{i});
end

% Reference values
refVal2 = 1.5;
refVal3 = 0.15;

% Initialize arrays for storing MAPE values for all datasets
all_MAPE_2 = cell(size(allData));
all_MAPE_3 = cell(size(allData));
all_firstBelow5_2 = zeros(size(allData));
all_firstBelow5_3 = zeros(size(allData));


% Initialize a figure
figure;

% Loop through all datasets for plotting
for i = 1:length(allData)
    % Extract current dataset
    currentData = allData{i};
    
    % Calculate MAPE for column 2 and 3
    [MAPE_2, firstBelow5_2] = calculateCumulativeMAPE(currentData(:,2), refVal2);
    [MAPE_3, firstBelow5_3] = calculateCumulativeMAPE(currentData(:,3), refVal3);
    
    % Subplot for column 2
    subplot(4,2,2*i-1); % Determine subplot position
    plot(currentData(:,1), MAPE_2);
    hold on;
    if ~isnan(firstBelow5_2)
        xline(currentData(firstBelow5_2, 1), 'r--', 'LineWidth', 2, 'Label', ['t=' num2str(currentData(firstBelow5_2, 1))]);
    end
    title(['Dataset ' char('A'+i-1) ' Column 2 MAPE Over Time']);
    xlabel('Time');
    ylabel('MAPE (%)');
    hold off;
    
    % Subplot for column 3
    subplot(4,2,2*i); % Determine subplot position
    plot(currentData(:,1), MAPE_3);
    hold on;
    if ~isnan(firstBelow5_3)
        xline(currentData(firstBelow5_3, 1), 'r--', 'LineWidth', 2, 'Label', ['t=' num2str(currentData(firstBelow5_3, 1))]);
    end
    title(['Dataset ' char('A'+i-1) ' Column 3 MAPE Over Time']);
    xlabel('Time');
    ylabel('MAPE (%)');
    hold off;
end

% Adjust layout to avoid label/title overlap
sgtitle('MAPE Over Time for Datasets A, B, C, D'); % Overall title




% Calculate MAPE for each dataset
for i = 1:length(allData)
    [all_MAPE_2{i}, all_firstBelow5_2(i)] = calculateCumulativeMAPE(allData{i}(:,2), refVal2);
    [all_MAPE_3{i}, all_firstBelow5_3(i)] = calculateCumulativeMAPE(allData{i}(:,3), refVal3);
    
    % Individual plots for each dataset
    figure;
    subplot(2,1,1);
    plot(allData{i}(:,1), all_MAPE_2{i});
    hold on;
    if ~isnan(all_firstBelow5_2(i))
        xline(allData{i}(all_firstBelow5_2(i), 1), 'r--', 'LineWidth', 2, 'Label', ['t=' num2str(allData{i}(all_firstBelow5_2(i), 1))]);
    end
    title(['Dataset ', char(96+i), ' Column 2 MAPE Over Time']);
    xlabel('Time');
    ylabel('MAPE (%)');
    hold off;
    
    subplot(2,1,2);
    plot(allData{i}(:,1), all_MAPE_3{i});
    hold on;
    if ~isnan(all_firstBelow5_3(i))
        xline(allData{i}(all_firstBelow5_3(i), 1), 'r--', 'LineWidth', 2, 'Label', ['t=' num2str(allData{i}(all_firstBelow5_3(i), 1))]);
    end
    title(['Dataset ', char(96+i), ' Column 3 MAPE Over Time']);
    xlabel('Time');
    ylabel('MAPE (%)');
    hold off;
end

% 对于所有数据集的第二列MAPE绘制在一个图中的不同小图窗
figure;
for i = 1:length(allData)
    subplot(length(allData),1,i); % 为每个数据集分配一个小图窗
    plot(allData{i}(:,1), all_MAPE_2{i});
    title(['Dataset ', char(64+i), ' Column 2 MAPE Over Time']);
    xlabel('Time');
    ylabel('MAPE (%)');
    if ~isnan(all_firstBelow5_2(i))
        hold on;
        xline(allData{i}(all_firstBelow5_2(i), 1), 'r--', 'LineWidth', 2, 'Label', ['t=' num2str(allData{i}(all_firstBelow5_2(i), 1))]);
        hold off;
    end
end

% 对于所有数据集的第三列MAPE绘制在一个图中的不同小图窗
figure;
for i = 1:length(allData)
    subplot(length(allData),1,i); % 为每个数据集分配一个小图窗
    plot(allData{i}(:,1), all_MAPE_3{i});
    title(['Dataset ', char(64+i), ' Column 3 MAPE Over Time']);
    xlabel('Time');
    ylabel('MAPE (%)');
    if ~isnan(all_firstBelow5_3(i))
        hold on;
        xline(allData{i}(all_firstBelow5_3(i), 1), 'r--', 'LineWidth', 2, 'Label', ['t=' num2str(allData{i}(all_firstBelow5_3(i), 1))]);
        hold off;
    end
end




%% 应用时间区间筛选和绘图
% 对于每个数据集，在全局时间区间内绘制MAPE

for i = 1:length(allData)
    % 对当前数据集应用时间区间筛选
    currentData = allData{i};
    timeFilter = currentData(:, 1) >= globalTimeStart & currentData(:, 1) <= globalTimeEnd;
    filteredData = currentData(timeFilter, :);

    % 重新计算筛选后数据的MAPE
    [filteredMAPE_2, filteredFirstBelow5_2] = calculateCumulativeMAPE(filteredData(:,2), refVal2);
    [filteredMAPE_3, filteredFirstBelow5_3] = calculateCumulativeMAPE(filteredData(:,3), refVal3);

    % 为筛选后的数据绘制MAPE图表，这里使用新的图形窗口以区分
    % 绘制第二列MAPE
    figure;
    plot(filteredData(:,1), filteredMAPE_2, 'b');
    title(['Filtered Dataset ' char(64+i) ' Column 2 MAPE Over Time']);
    xlabel('Time');
    ylabel('MAPE (%)');
    if ~isnan(filteredFirstBelow5_2)
        hold on;
        xline(filteredData(filteredFirstBelow5_2, 1), 'r--', 'LineWidth', 2, 'Label', ['5% Error @ t=' num2str(filteredData(filteredFirstBelow5_2, 1))]);
        hold off;
    end

    % 绘制第三列MAPE
    figure;
    plot(filteredData(:,1), filteredMAPE_3, 'b');
    title(['Filtered Dataset ' char(64+i) ' Column 3 MAPE Over Time']);
    xlabel('Time');
    ylabel('MAPE (%)');
    if ~isnan(filteredFirstBelow5_3)
        hold on;
        xline(filteredData(filteredFirstBelow5_3, 1), 'r--', 'LineWidth', 2, 'Label', ['5% Error @ t=' num2str(filteredData(filteredFirstBelow5_3, 1))]);
        hold off;
    end
end
