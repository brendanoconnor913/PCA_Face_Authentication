function [frrresults, steps] = PCA_Mode1()

ddir = dir('./orl_faces');
data = []; % Matrix to store all of our images
inc = 1;
% Read through all of the image directories and get the first half of each
% directory (5 of the 10 images)
for k = 3:length(ddir) % discard '.' and '..'
    if (ddir(k).isdir)
        fname = strcat('orl_faces/',ddir(k).name);
        disp(fname);
        imds = imageDatastore(fname);
        % load all images into matrix
        for i = 1:(length(imds.Files)/2)
            % Scale the image and add to our matrix of all images
            m = imresize(double(readimage(imds,i)), .2);
            l = reshape(m, [437,1]);
            data(:,inc) = l;
            inc = inc + 1;
        end
    end
end

[r,c] = size(data);
% Compute the mean of each image
m = mean(data);
% Subtract the mean from each image [Centering the data]
d = data-repmat(m,r,1);

% Compute the covariance matrix (co)
co = d*d';

% Compute the eigen values and eigen vectors of the covariance matrix
[eigvector,eigvl] = eig(co);

% Sort the eigen vectors according to the eigen values
eigvalue = diag(eigvl);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

% Compute the number of eigen values that greater than zero (you can select any threshold)
count1=0;
for i=1:size(eigvalue,1)
    if(eigvalue(i)>0)
        count1=count1+1;
    end
end

% Calculate proportion of variance from non zero eigenvalues
% and get input from user on how many vectors to use
nonzeig = eigvalue(1:count1);
if (length(nonzeig) > 1)
    results = [];
    tempvecs = [];
    sumnz = sum(nonzeig);
    for i = 1:(length(nonzeig))
        tempvecs(end+1) = nonzeig(i);
        sumt = sum(tempvecs);
        r = sumt/sumnz;
        results(end+1) = r;
    end
    numvec = (1:length(results));
    plot(numvec, results);
    out = 'Enter number of vectors to use: ';
    num = input(out);
end

% Transform data to new eigenspace
vec = eigvector(:,1:num);
x = vec'*d;

% Load appropriate test data into matrix
tdata = [];
inc = 1;
for k = 3:length(ddir) % discard '.' and '..'
    if (ddir(k).isdir)
        fname = strcat('orl_faces/',ddir(k).name);
        disp(fname);
        imds = imageDatastore(fname);
        % load all images into 
        for i = (length(imds.Files)/2)+1:(length(imds.Files))
            m = imresize(double(readimage(imds,i)), .2);
            l = reshape(m, [437,1]);
            tdata(:,inc) = l;
            inc = inc + 1;
        end
    end
end

% Transform test data into eigenspace
[r,c] = size(tdata);
m = mean(tdata);
tdata = tdata-repmat(m,r,1);
newtest = vec'*tdata;

% Evaluate false reject rate at various distance rejection thresholds
frrresults = [];
steps = [];
p = 700;
while (p > 0)
    predictlabels = [];
    actuallabels = [];
    class = 0;
    frrnum = 0;
    rejects = [];
    accepts = [];

    for i=1:size(newtest,2)
        % Compute distance from i'th test image to all training data
        alldata = newtest(:,i)';
        alldata(2:size(x,2)+1,:) = x';
        dist = pdist(alldata);
        [a,b] = min(dist(:,1:size(x,2)));

        % If distance greater than threshold, reject
        if (a < p)
            predictlabels(end+1) = ceil(b/5);
        else
            predictlabels(end+1) = -1;
        end

        if ((mod(i,5)) == 1)
            class = class + 1;
        end
        actuallabels(end+1) = class;
        % If predicted label doesn't match known class add to frr sum
        if (class ~= predictlabels(end))
            frrnum = frrnum + 1;
            rejects(end+1) = a;
        else
            accepts(end+1) = a;
        end
    end
    frr = frrnum / 200;
    frrresults(end+1) = frr;
    steps(end+1) = p;
    p = p - 5;
end

