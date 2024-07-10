function data = readFormattedDatFile(filePath)
    % 打开文件
    fileId = fopen(filePath, 'r');
    
    % 检查文件是否成功打开
    if fileId == -1
        error('文件无法打开：%s', filePath);
    end
    
    % 初始化存储数据的容器
    data = [];
    
    % 按行读取和解析文件
    while ~feof(fileId)
        line = fgetl(fileId); % 读取一行
        if ischar(line)
            % 使用正则表达式提取括号内的数值
            nums = regexp(line, '\[(.*?)\]', 'tokens');
            if ~isempty(nums)
                % 转换字符串为数字
                nums = str2num(char(nums{1})); % 可能需要根据实际格式调整
                data = [data; nums]; % 添加到数据容器
            end
        end
    end
    
    % 关闭文件
    fclose(fileId);
end
