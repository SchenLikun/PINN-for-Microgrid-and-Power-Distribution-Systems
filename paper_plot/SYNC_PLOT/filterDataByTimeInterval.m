function filteredData = filterDataByTimeInterval(data, timeStart, timeEnd)
    % 筛选出在指定时间区间内的数据
    timeFilter = data(:, 1) >= timeStart & data(:, 1) <= timeEnd;
    filteredData = data(timeFilter, :);
end
