function Plot_data()

    clc; clear; close all;
    
    global allData table realValue realLine ax table_data;
    allData = struct();
    realValue = [];
    realLine = [];
    table_data = {};

    fig = figure('Name', 'Comparison (interactive_v9_final3)', 'NumberTitle', 'off', ...
        'Position', [100, 100, 1200, 700]);

    uicontrol(fig, 'Style', 'pushbutton', 'String', 'Add File', ...
        'Position', [20, 660, 100, 30], 'Callback', @add_files);
    uicontrol(fig, 'Style', 'pushbutton', 'String', 'Remove data', ...
        'Position', [140, 660, 130, 30], 'Callback', @remove_selected_files);
    uicontrol(fig, 'Style', 'pushbutton', 'String', 'Real Value Set', ...
        'Position', [260, 660, 100, 30], 'Callback', @modify_real_value);
    uicontrol(fig, 'Style', 'pushbutton', 'String', 'Time', ...
        'Position', [380, 660, 130, 30], 'Callback', @modify_x_range); % 新增按钮

    % 创建表格
    table = uitable(fig, 'Data', table_data, 'ColumnName', ...
        {'choose', 'dir', 'filename', 'signal', 'fileNo.', 'dataNo.'}, ...
        'ColumnEditable', [true, false, false, false, false, false], ...
        'Position', [20, 350, 400, 300], 'CellEditCallback', @update_plot);

    % 创建绘图区域
    ax = axes(fig, 'Position', [0.45, 0.15, 0.5, 0.75]);
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    title(ax, 'Comparison');
    xlabel(ax, 'Time'); ylabel(ax, 'Predict value');
    
    update_plot();
end

%% ============ 主要函数 =============

function add_files(~, ~)
    global allData table table_data;
    [files, pathStr] = uigetfile('*.mat', 'choose .mat file', 'MultiSelect', 'on');
    if isequal(files, 0), return; end
    if ischar(files), files = {files}; end

    for k = 1:length(files)
        fullPath = fullfile(pathStr, files{k});
        data = load(fullPath);
        t = data.timePoints;
        dCell = data.dataCell;
        maxCols = max(cellfun(@length, dCell));
        parts = strsplit(pathStr, filesep);
        parentDir = parts{end-1};
        idx = length(allData) + 1;
        allData(idx).fileName = files{k};
        allData(idx).parentDir = parentDir;
        allData(idx).timePoints = t;
        allData(idx).dataCell = dCell;
        allData(idx).maxCols = maxCols;

        td = table.Data;
        for c = 1:maxCols
            td{end+1, 1} = false;  
            td{end, 2} = parentDir;
            td{end, 3} = files{k};
            td{end, 4} = ['col ' num2str(c)];
            td{end, 5} = idx;
            td{end, 6} = c;
        end
        table.Data = td;
    end
    update_plot();
end

function remove_selected_files(~, ~)
    global table table_data;
    if isempty(table.Data), return; end
    d = table.Data;
    sel = find(cell2mat(d(:,1)));
    d(sel, :) = [];
    table_data = d;
    table.Data = d;
    update_plot();
end

function modify_real_value(~, ~)
    global realValue;
    prompt = {'Real Value Set：'};
    answer = inputdlg(prompt, 'Real Value Set', [1, 50]);
    if isempty(answer), return; end
    val = str2double(answer{1});
    if isnan(val), return; end
    realValue = val;
    update_plot();
end

function modify_x_range(~, ~)
    global ax allData table;

    if isempty(table.Data)
        msgbox('Plz load data！', 'error', 'error');
        return;
    end

    prompt = {'e.g. "30-50"：'};
    answer = inputdlg(prompt, 'Time', [1, 50]);
    if isempty(answer), return; end

    rangeStr = strsplit(strtrim(answer{1}), '-');  
    if length(rangeStr) ~= 2
        msgbox('error, plz input like 0-100', 'error', 'error');
        return;
    end

    percentMin = str2double(rangeStr{1}) / 100;
    percentMax = str2double(rangeStr{2}) / 100;

    if isnan(percentMin) || isnan(percentMax) || percentMin < 0 || percentMax > 1 || percentMin >= percentMax
        msgbox('error', 'fucking error', 'error');
        return;
    end

    minTime = inf;
    maxTime = -inf;
    for i = 1:length(allData)
        if isempty(allData(i).timePoints)
            continue;
        end
        validTime = allData(i).timePoints(~isnan(allData(i).timePoints));  
        if isempty(validTime)
            continue;
        end
        minTime = min(minTime, min(validTime));
        maxTime = max(maxTime, max(validTime));
    end

    if isinf(minTime) || isinf(maxTime) || minTime == maxTime
        msgbox('fucking error', 'error', 'error');
        return;
    end

    newXMin = minTime + (maxTime - minTime) * percentMin;
    newXMax = minTime + (maxTime - minTime) * percentMax;

    if newXMin >= newXMax
        msgbox('NoneValid', 'warning', 'warn');
        xlim(ax, [minTime, maxTime]);  
    else
        xlim(ax, [newXMin, newXMax]);  
    end
end

function update_plot(~, ~)
    global ax allData table realValue realLine;
    
    cla(ax);  

    if ~isempty(realValue)
        realLine = yline(ax, realValue, '--k', 'Real Value', 'LineWidth', 2);
        realLine.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end

    d = table.Data;
    if isempty(d)
        drawnow;
        return;
    end

    sel = find(cell2mat(d(:,1)));
    if isempty(sel)
        drawnow;
        return;
    end

    colors = lines(length(sel));
    legendEntries = {};

    for i = 1:length(sel)
        row = sel(i);
        fileIdx = d{row,5};
        colIdx = d{row,6};

        t = allData(fileIdx).timePoints;
        y = cellfun(@(c) c(colIdx), allData(fileIdx).dataCell, 'UniformOutput', false);
        y = cell2mat(y);

        hl = plot(ax, t, y, '-', 'LineWidth', 1.5, 'Color', colors(i, :));
        legendEntries{end+1} = d{row,3};
    end

    legend(ax, legendEntries, 'Location', 'best');
    grid(ax, 'on'); box(ax, 'on');
    drawnow;
end
