function [Forward, mask] = feedforward(Input,Weight,Bias,dropout)

out = Weight*Input + Bias;
Forward = 1./(1+exp(-out));
compare = rand(size(Forward));
mask = (compare > dropout);
Forward = Forward.*mask;
   
end
    