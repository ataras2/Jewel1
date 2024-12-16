; Program to find low redundancy tesellating arrays 
;
;
; This version for simple circular pupil 
;  try to find 7 patterns; 4 with 7 segs and 3 with 6 segs = 46 total
;  
; Details:

; This version:  face center grid

nholes=7
tha=intarr(2,nholes)
rcount=0
nmasks=7  ; find this many sets of nholes masks
available_segments = 46

ignore_short = 0 ; don't worry about baselines shorter than this for redundancy

unused_matrix_slots = nholes*nmasks - available_segments
n_filled_cols = nmasks - unused_matrix_slots

good_threshold=10

outfile = 'find46_4x7h3x6h.dat'

;openw,1,outfile
;printf,1,'This file has indices for 4 * 7 hole + 3 x 6h mask'
;close,1


idim=9           ; Dimensions of Grid
jdim=9



; Make up array of forbidden locations for holes.
ok=[[1,1,1,1,1,0,0,0,0], $
    [1,1,1,1,1,1,0,0,0], $
    [1,1,1,0,1,1,1,0,0], $
    [1,1,0,0,0,0,1,1,0], $
    [1,1,0,0,0,0,0,1,1], $
    [0,1,1,0,0,0,0,1,1], $
    [0,0,1,1,1,0,1,1,1], $
    [0,0,0,1,1,1,1,1,1], $
    [0,0,0,0,1,1,1,1,1]] 
ok_aa=ok


tryoveragain:

; For a starting point, randomly suffle available slots into subartrays (nholes x nmasks)
good_slots = where(ok eq 1)
total_holes = available_segments
shuffle_start_ix = sort(randomu(seed,total_holes))
shuffle_start = good_slots(shuffle_start_ix)

all_holes = intarr(nmasks,nholes)

all_holes[0:available_segments-1] = shuffle_start

; find redundancy of starting point ...

best_redun = 0
for mask=0,nmasks-1 do begin 
    thismask = reform(all_holes(mask,*))

    if(mask ge n_filled_cols) then thismask = thismask[0:nholes-2]

    maskgrid = transpose( [ [thismask mod idim], [thismask/idim] ] )
    best_redun += test_mask_nr(maskgrid,idim=idim,jdim=jdim)
endfor

best_pattern = all_holes

for i=0,1000000 do begin

    shuffle_mask = (sort(randomu(seed,nmasks)))[0:1]
    shuffle_holes = fix(randomu(seed,2) * nholes)

    if( (shuffle_mask[0] ge n_filled_cols) and (shuffle_holes[0] eq nholes-1) ) then goto,not_this_one
    if( (shuffle_mask[1] ge n_filled_cols) and (shuffle_holes[1] eq nholes-1) ) then goto,not_this_one

    all_holes = best_pattern
    all_holes[shuffle_mask,shuffle_holes] = best_pattern[reverse(shuffle_mask),reverse(shuffle_holes)]
    this_redun = 0
    for mask=0,nmasks-1 do begin 
       thismask = reform(all_holes(mask,*))

       if(mask ge n_filled_cols) then thismask = thismask[0:nholes-2]

       maskgrid = transpose( [ [thismask mod idim], [thismask/idim] ] )
       this_redun += test_mask_nr(maskgrid,idim=idim,jdim=jdim)
   endfor
    
    if(this_redun le best_redun) then begin
        best_pattern = all_holes
        best_redun = this_redun
        print,i,best_redun,' - 2'
    endif

    not_this_one:

    shuffle_mask = (sort(randomu(seed,nmasks)))[0:2]
    shuffle_holes = fix(randomu(seed,3) * nholes)

    if( (shuffle_mask[0] ge n_filled_cols) and (shuffle_holes[0] eq nholes-1) ) then goto,not_this_two
    if( (shuffle_mask[1] ge n_filled_cols) and (shuffle_holes[1] eq nholes-1) ) then goto,not_this_two
    if( (shuffle_mask[2] ge n_filled_cols) and (shuffle_holes[2] eq nholes-1) ) then goto,not_this_two

    all_holes = best_pattern
    all_holes[shuffle_mask,shuffle_holes] = best_pattern[shift(shuffle_mask,1),shift(shuffle_holes,1)]
    this_redun = 0
    for mask=0,nmasks-1 do begin 
        thismask = reform(all_holes(mask,*))

        if(mask ge n_filled_cols) then thismask = thismask[0:nholes-2]

        maskgrid = transpose( [ [thismask mod idim], [thismask/idim] ] )
        this_redun += test_mask_nr(maskgrid,idim=idim,jdim=jdim)
    endfor
    
    if(this_redun le best_redun) then begin
        best_pattern = all_holes
        best_redun = this_redun
        print,i,best_redun,' - 3'
    endif

    not_this_two:

endfor

if(best_redun le good_threshold) then begin
   openu,1,outfile,/append
   rcount=rcount+1
   print,rcount,' ',best_redun
   printf,1,rcount,best_redun
   print,best_pattern
   printf,1,best_pattern
   close,1
   good_threshold = best_redun          ; push down threshold

endif

goto,tryoveragain


end

