% 定义函数计算累积L2相对误差
function errors = cumulativeL2RelativeError(data, ref)
    errors = zeros(size(data));
    for i = 1:length(data)
        cumulativeSum = sum((data(1:i) - ref).^2);
        errors(i) = sqrt(cumulativeSum / (i * ref^2));
    end
end