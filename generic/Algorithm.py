#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:24:44 2019

@author: edward
"""

def array_overlap(nums1, nums2):
    n_start = max([nums1[0],  nums2[0]])
    n_end   = min([nums1[-1], nums2[-1]])
    
    print(n_start)
    print(n_end)
    
    nums1 = [n for n in nums1 if n >= n_start and n <= n_end]
    nums2 = [n for n in nums2 if n >= n_start and n <= n_end]

    return nums1, nums2



if __name__ == '__main__':
    nums1 = [1,2, 3, 5, 6]
    nums2 = [3, 4, 7]
    
    print(array_overlap(nums1, nums2))

