% Define the file path
filePath = 'SYNC_variables_0316_TranAll.dat';  % update the path if the file is in a different directory

% Open the file for reading
fileID = fopen(filePath, 'r');

% Check if the file was opened successfully
if fileID == -1
    error('File cannot be opened');
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

% Check if any data was collected
if ~isempty(data) && size(data, 2) == 3
    % Assign the data to the variables
    time = data(:, 1);        % Time data
    variable1 = data(:, 2);    % First variable
    variable2 = data(:, 3);    % Second variable
    disp('Data loaded successfully.');
else
    fprintf('Data array size: %dx%d\n', size(data, 1), size(data, 2));
    error('Data does not have three columns as expected.');
end

% Display first few rows to verify
disp(data(1:min(end, 5), :));
