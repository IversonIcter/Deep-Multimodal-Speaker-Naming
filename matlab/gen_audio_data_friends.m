%%% generate audio samples (all in one, i.e. no split for training/test)

close all;
clc;clear;

folder_base = '/home/wyj/dataset/Friends/Friends.S05E05/speaking-audio';
%folder_base = '/home/wyj/dataset/Friends-Audio';

names = {'chandler', 'joey', 'monica', 'phoebe', 'rachel','ross'}

% start here...
samples_all = [];
labels_all = [];
for i=1:6
	sprintf('%s/%s_merged_all.wav', folder_base, names{i});
    audioFile = sprintf('%s/%s.wav', folder_base, names{i});
    %audioFile = sprintf('%s/%s_merged_all.wav', folder_base, names{i});
    audioData = miraudio(audioFile);
    
    % split into different frames, 20ms as one, half overlapping
    frames = mirframe(audioData, 'Length', 0.02, 's'); 
    
    % compute 25d mfcc + delta + delta, all 75d in total
    mfcc = mirgetdata(mirmfcc(frames, 'Rank', 1:25));
    mfcc_d1 = mirgetdata(mirmfcc(frames, 'Rank', 1:25, 'Delta', 1));
    mfcc_d2 = mirgetdata(mirmfcc(frames, 'Rank', 1:25, 'Delta', 2));
    
    % merge them together
    samples = [mfcc(:, 9:end); mfcc_d1(:, 5:end); mfcc_d2];   
    
    % normalize to zero mean and unit variance, over 300 frames
    sz_sliding = 300;
    num_sliding = size(samples, 2) / sz_sliding;
    for j=1:ceil(num_sliding)
        idx_start = (j-1)*sz_sliding+1;
        idx_end = j*sz_sliding;
        if j==ceil(num_sliding)
            idx_end = size(samples, 2);
        end
        
        samples_roi = samples(:, idx_start:idx_end);

        % zero mean and unit variance to roi
        samples_roi = samples_roi - mean(samples_roi(:));
        samples_roi = samples_roi / std(samples_roi(:));

        % write back
        samples(:, idx_start:idx_end) = samples_roi;
    end
    
%     % save to disk
%     save(sprintf('%s/audio_samples_%d.mat', folder_base, i), 'samples', '-v7.3'); 

    % label info
    labels = zeros(6, size(mfcc_d2, 2));
    labels(i, :) = 1;

    % merge to final mat
    samples_all = [samples_all samples];
    labels_all = [labels_all labels];
end

% save to disk
sample = samples_all;
tag = labels_all;
save(sprintf('%s/audio_samples.mat', folder_base), 'sample', 'tag', '-v7.3'); 
