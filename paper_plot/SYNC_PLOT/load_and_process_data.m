function data = load_and_process_data(filePath)
    % Open the file for reading
    fileID = fopen(filePath, 'r');
    
    % Check if the file was opened successfully
    if fileID == -1
        error(['File cannot be opened: ', filePath]);
    end
    
    % Initialize the array to hold the numeric data
    data = [];
    
    % Read the file line by line
    lineNumber = 0;
    while ~feof(fileID)
        lineNumber = lineNumber + 1;
        line = fgetl(fileID);
        if ischar(line)  % Check if the line is a character array
            % Use a regular expression to match numbers in scientific notation
            nums = regexp(line, '(\-?\d+\.?\d*(e[\+\-]?\d+)?)', 'match');
            
            % Check if there are exactly three numbers on the line
            if length(nums) == 3
                % Convert the matched strings to numbers and append to data
                data = [data; str2double(nums)];
            else
                fprintf('Line %d does not have three numbers: %s\n', lineNumber, line);
                % Break the loop on the first error to prevent flooding the console
                break;
            end
        end
    end
    
    % Close the file
    fclose(fileID);
    
    % Check if data was collected correctly
    if isempty(data) || size(data, 2) ~= 3
        fprintf('Data array size: %dx%d\n', size(data, 1), size(data, 2));
        error(['Data from file ', filePath, ' does not have three columns as expected.']);
    end
end



% Now you have the variables a, b, c, and d containing the data from each file
