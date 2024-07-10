% Function to calculate cumulative MAPE and find the first index below 5% error
function [mapeValues, firstBelow5PercentIndex] = calculateCumulativeMAPE(predictedValues, trueValue)
    mapeValues = zeros(length(predictedValues), 1);
    firstBelow5PercentIndex = NaN; % Initialize with NaN indicating not found
    
    for i = 1:length(predictedValues)
        % Calculate MAPE up to the i-th point
        % mapeValues(i) = mean(abs((predictedValues(1:i) - trueValue) / trueValue)) * 100;
        mapeValues(i) = abs((predictedValues(i) - trueValue) / trueValue) * 100;
        % Check if this is the first instance of MAPE being below 5%
        if isnan(firstBelow5PercentIndex) && mapeValues(i) <= 5
            firstBelow5PercentIndex = i;
        end
    end
end
