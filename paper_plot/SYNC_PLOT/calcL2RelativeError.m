% Function to calculate L2 relative error
function l2Error = calcL2RelativeError(data, refVal)
    numerator = sum((data - refVal).^2);
    denominator = sum(refVal^2) * numel(data);
    l2Error = sqrt(numerator / denominator);
end
