function [err] = crossentropy(label,output)

err = 0;
for i=1:1:size(output,1)
    err = err - log(output(i,label(i)+1));    
end

end