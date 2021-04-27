%% reading one channel from EDF
% Laseeb v1.1 - 2014-05-06
% fname = ´filename.edf'
% channel = 'C4A1' - Case sensity, substring find
% b return channel
% sf channel sampling frequency
% uses edfread.m
function [b,sf] = getedf(fname,channel)
    [hdr,rcd] = edfread(fname);
	chn = 0;
	for k = 1:hdr.ns
        if strfind(hdr.label{k},channel),
            chn = k;
        end
    end
    if chn == 0
        hdr.label %display available channels
        error('getedf -> channel = %s not found =',channel);
        exit;
    end
    sf = hdr.samples(chn)/hdr.duration;
    b = rcd(chn,1:end);
end

