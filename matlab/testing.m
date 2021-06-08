clear; clc;

matrix = zeros(243,5,'uint32');

for i = 0:242
    cos = dec2base(i,3);
    if strlength(cos) == 1
        cos = strcat('0000',cos);
    elseif strlength(cos) == 2
        cos = strcat('000',cos);
    elseif strlength(cos) == 3
        cos = strcat('00',cos);
    elseif strlength(cos) == 4
        cos = strcat('0',cos);
    end
    for j = 1:5
        matrix(i+1,j) = str2double(cos(j));
    end
end
matrix