function [Forward] = softmax(Input,Weight,Bias)

out = Weight*Input + Bias;
Forward = exp(out)/sum(exp(out));
% Forward(isnan(Forward)) = 1;
% Forward = Forward + realmin;
% Forward = Forward/sum(Forward);
end