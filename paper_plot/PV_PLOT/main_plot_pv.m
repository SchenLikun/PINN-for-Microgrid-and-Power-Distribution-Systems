clear;
clc;


% 定义文件路径
filePaths = {
    'PI_variables_ModelSave_0306_4obs.dat',
    'PI_variables_swish_transIn_ObAll_0226.dat',
    'PI_variables_swish_transOut_ObAll_0226.dat',
    'PI_variables_tanh_transNone_0224.dat'
};


% 定义每列数据对应的颜色
colors = {
    '#f2931e', % 橙色
    '#ae4132', % 暗红色
    '#10739e', % 深蓝色
    '#23445d'  % 藏蓝色
};

xStart = 1; % 起始值
xEnd = 1e4; % 结束值
% 每隔n个数据点计算误差
n = 1000; 
% 自定义横坐标，假设每隔n个数据点有一个实际的数据点标签
xLabels = xStart:n:xEnd; % 此数组应与实际数据点相匹配
% 真实值
trueValues = [0.025, 5.0, 0.025, 0.5];


% 初始化数据存储容器
dataSets = cell(length(filePaths), 1);

% 读取数据文件
for i = 1:length(filePaths)
    dataSets{i} = readFormattedDatFile(filePaths{i});
end

% 创建包含四个子图的图形，每个子图对应一个文件的四列数据
figure('Name', 'Subplots for Each Data File', 'NumberTitle', 'off');
for i = 1:length(dataSets)
    subplot(2, 2, i); % 分为2x2子图布局
    hold on;
    for j = 1:4
        plot(dataSets{i}(:,j), 'Color', colors{j}, 'LineWidth', 1.5);
    end
    hold off;
    
    ylim([0 5.5]); % 设置y轴范围
    box on; % 开启边框
    % 添加灰色虚线参考
    yline(0.025, '--', 'Color', '#000000', 'LineWidth', 1);
    yline(0.5, '--', 'Color', '#000000', 'LineWidth', 1);
    yline(5, '--', 'Color', '#000000', 'LineWidth', 1);
    
    title(['Graph for ', filePaths{i}]);
    xlabel('X-axis Label');
    ylabel('Y-axis Label');
    legend({'Kp', 'Ki', 'IKp', 'IKi'}, 'Location', 'best');
end

% 为每个数据文件创建单独的图形
for i = 1:length(dataSets)
    figure('Name', ['Individual Plot for ', filePaths{i}], 'NumberTitle', 'off');
    hold on;
    for j = 1:4
        plot(dataSets{i}(:,j), 'Color', colors{j}, 'LineWidth', 1.5);
    end
    hold off;
    
    ylim([0 5.5]);
    box on;
    % 添加灰色虚线参考
    yline(0.025, '--', 'Color', '#000000', 'LineWidth', 1);
    yline(0.5, '--', 'Color', '#000000', 'LineWidth', 1);
    yline(5, '--', 'Color', '#000000', 'LineWidth', 1);
    
    title(['Graph for ', filePaths{i}]);
    xlabel('X-axis Label');
    ylabel('Y-axis Label');
    legend({'Kp', 'Ki', 'IKp', 'IKi'}, 'Location', 'best');
end



%% 区间绘图
% 为每个数据文件创建单独的图形，并只在指定横坐标区间绘制
% 区间绘图部分
for i = 1:length(dataSets)
    figure('Name', ['Individual Plot for ', filePaths{i}], 'NumberTitle', 'off');
    hold on;
    for j = 1:4
        % 仅选取横坐标区间 [xStart, xEnd] 内的数据
        % 确保 xStart 和 xEnd 在数据长度内
        plot(xStart:xEnd, dataSets{i}(xStart:xEnd, j), 'Color', colors{j}, 'LineWidth', 1.5);
    end
    hold off;
    
    xlim([xStart xEnd]); % 设置x轴范围
    ylim([0 5.5]); % 设置y轴范围
    box on; % 开启边框
    % 添加黑色虚线参考
    yline(0.025, '--', 'Color', '#000000', 'LineWidth', 1);
    yline(0.5, '--', 'Color', '#000000', 'LineWidth', 1);
    yline(5, '--', 'Color', '#000000', 'LineWidth', 1);
    
    title(['Graph for ', filePaths{i}]);
    xlabel('X-axis Label');
    ylabel('Y-axis Label');
    legend({'Kp', 'Ki', 'IKp', 'IKi'}, 'Location', 'best');
end


% 创建包含四个子图的图形，每个子图对应一个文件的四列数据
figure('Name', 'Subplots for Each Data File with Custom X-axis', 'NumberTitle', 'off');
for i = 1:length(dataSets)
    subplot(2, 2, i); % 分为2x2子图布局
    hold on;
    for j = 1:4
        xValues = xStart:xEnd; % 创建横坐标向量，假设数据点是连续且等间隔的
        % 根据自定义横坐标范围绘制每列数据
        plot(xValues, dataSets{i}(xStart:xEnd, j), 'Color', colors{j}, 'LineWidth', 1.5);
    end
    hold off;
    
    xlim([xValues(1), xValues(end)]); % 设置自定义横坐标范围
    ylim([0 5.5]); % 设置y轴范围
    box on; % 开启边框
    % 添加灰色虚线参考
    yline(0.025, '--', 'Color', '#000000', 'LineWidth', 1);
    yline(0.5, '--', 'Color', '#000000', 'LineWidth', 1);
    yline(5, '--', 'Color', '#000000', 'LineWidth', 1);
    
    title(['Graph for ', filePaths{i}]);
    xlabel('X-axis Label');
    ylabel('Y-axis Label');
    legend({'Kp', 'Ki', 'IKp', 'IKi'}, 'Location', 'best');
end



%% 误差部分
% 读取数据文件并计算误差
errorData = cell(length(filePaths), 1);
for i = 1:length(filePaths)
    % 假设 readFormattedDatFile 返回一个15001x4的矩阵
    data = readFormattedDatFile(filePaths{i});
    
    % 初始化误差矩阵
    errorMatrix = zeros(ceil(size(data, 1) / n), 4);
    
    for j = 1:4
        for k = 1:n:size(data, 1)
            % 计算每隔n个数据点的误差
            endIndex = min(k + n - 1, size(data, 1));
            errorMatrix(ceil(k/n), j) = mean(abs(data(k:endIndex, j) - trueValues(j))/trueValues(j)*100 );
        end
    end
    
    errorData{i} = errorMatrix;
end

% 绘制误差柱状图
figure('Name', 'Training Error Visualization', 'NumberTitle', 'off');
for i = 1:length(errorData)
    subplot(2, 2, i);
    barData = errorData{i};
    numBars = size(barData, 1);
    
    hold on;
    for j = 1:4
        % 创建累积的误差数据
        if j == 1
            bar(1:numBars, barData(:, j), 'FaceColor', colors{j}, 'EdgeColor', colors{j});
        else
            bar(1:numBars, sum(barData(:, 1:j), 2), 'FaceColor', colors{j}, 'EdgeColor', colors{j});
        end
    end
    hold off;
    
    title(['Error for ', filePaths{i}]);
    xlabel('Data Point Interval');
    ylabel('Error');
    xlim([0 numBars + 1]);
    % 可以调整更多的图形设置...
end



% 绘制误差柱状图
figure('Name', 'Training Error Visualization', 'NumberTitle', 'off');
for i = 1:length(errorData)
    subplot(2, 2, i);
    barData = errorData{i};
    numBars = size(barData, 1);
    
    % 绘制堆叠条形图
    b = bar(1:numBars, barData, 'stacked'); % 使用 'stacked' 选项来堆叠
    
    % 为每部分设置颜色
    for j = 1:4
        b(j).FaceColor = colors{j};
        b(j).EdgeColor = colors{j};
    end
    
    title(['Error for ', filePaths{i}]);
    xlabel('Data Point Interval');
    ylabel('Error (%)');
    xlim([0 numBars + 1]);
    legend({'Kp', 'Ki', 'IKp', 'IKi'}, 'Location', 'best', 'Interpreter', 'none');
end


% 绘制自定义横坐标范围的误差柱状图
figure('Name', 'Custom Training Error Visualization', 'NumberTitle', 'off');
for i = 1:length(errorData)
    subplot(2, 2, i);
    % 选择与自定义横坐标相匹配的误差数据的子集
    barData = errorData{i}(ceil(xStart/n):ceil(xEnd/n), :);
    numBars = length(xLabels);
    
    % 绘制堆叠条形图
    b = bar(1:numBars, barData, 'stacked'); % 使用 'stacked' 选项来堆叠
    
    % 为每部分设置颜色
    for j = 1:4
        b(j).FaceColor = colors{j};
        b(j).EdgeColor = colors{j};
    end
    
    % 设置自定义横坐标
    % set(gca, 'xtick', 0:numBars-1, 'xticklabel', xLabels);
    
    title(['Error for ', filePaths{i}]);
    xlabel('Data Point');
    ylabel('Error (%)');
    % xlim([0 numBars + 1]);
    legend({'Kp', 'Ki', 'IKp', 'IKi'}, 'Location', 'best', 'Interpreter', 'none');
    box on;
end



