  .text
  .globl _ZNSt11__timepunctIwEC1EPSt17__timepunct_cacheIwEj
  .type _ZNSt11__timepunctIwEC1EPSt17__timepunct_cacheIwEj, @function

#! file-offset 0xf61c0
#! rip-offset  0xb61c0
#! capacity    192 bytes

# Text                                                      #  Line  RIP      Bytes  Opcode              
._ZNSt11__timepunctIwEC1EPSt17__timepunct_cacheIwEj:        #        0xb61c0  0      OPC=<label>         
  pushq %rbx                                                #  1     0xb61c0  1      OPC=pushq_r64_1     
  xorl %eax, %eax                                           #  2     0xb61c1  2      OPC=xorl_r32_r32    
  movl %edi, %ebx                                           #  3     0xb61c3  2      OPC=movl_r32_r32    
  subl $0x10, %esp                                          #  4     0xb61c5  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                                           #  5     0xb61c8  3      OPC=addq_r64_r64    
  testl %edx, %edx                                          #  6     0xb61cb  2      OPC=testl_r32_r32   
  movl %ebx, %ebx                                           #  7     0xb61cd  2      OPC=movl_r32_r32    
  movl $0x1003c148, (%r15,%rbx,1)                           #  8     0xb61cf  8      OPC=movl_m32_imm32  
  setne %al                                                 #  9     0xb61d7  3      OPC=setne_r8        
  nop                                                       #  10    0xb61da  1      OPC=nop             
  nop                                                       #  11    0xb61db  1      OPC=nop             
  nop                                                       #  12    0xb61dc  1      OPC=nop             
  nop                                                       #  13    0xb61dd  1      OPC=nop             
  nop                                                       #  14    0xb61de  1      OPC=nop             
  nop                                                       #  15    0xb61df  1      OPC=nop             
  movl %ebx, %ebx                                           #  16    0xb61e0  2      OPC=movl_r32_r32    
  movl %esi, 0x8(%r15,%rbx,1)                               #  17    0xb61e2  5      OPC=movl_m32_r32    
  movl %ebx, %ebx                                           #  18    0xb61e7  2      OPC=movl_r32_r32    
  movl %eax, 0x4(%r15,%rbx,1)                               #  19    0xb61e9  5      OPC=movl_m32_r32    
  nop                                                       #  20    0xb61ee  1      OPC=nop             
  nop                                                       #  21    0xb61ef  1      OPC=nop             
  nop                                                       #  22    0xb61f0  1      OPC=nop             
  nop                                                       #  23    0xb61f1  1      OPC=nop             
  nop                                                       #  24    0xb61f2  1      OPC=nop             
  nop                                                       #  25    0xb61f3  1      OPC=nop             
  nop                                                       #  26    0xb61f4  1      OPC=nop             
  nop                                                       #  27    0xb61f5  1      OPC=nop             
  nop                                                       #  28    0xb61f6  1      OPC=nop             
  nop                                                       #  29    0xb61f7  1      OPC=nop             
  nop                                                       #  30    0xb61f8  1      OPC=nop             
  nop                                                       #  31    0xb61f9  1      OPC=nop             
  nop                                                       #  32    0xb61fa  1      OPC=nop             
  callq ._ZNSt6locale5facet13_S_get_c_nameEv                #  33    0xb61fb  5      OPC=callq_label     
  movl %ebx, %ebx                                           #  34    0xb6200  2      OPC=movl_r32_r32    
  movl %eax, 0x10(%r15,%rbx,1)                              #  35    0xb6202  5      OPC=movl_m32_r32    
  xorl %esi, %esi                                           #  36    0xb6207  2      OPC=xorl_r32_r32    
  movl %ebx, %edi                                           #  37    0xb6209  2      OPC=movl_r32_r32    
  nop                                                       #  38    0xb620b  1      OPC=nop             
  nop                                                       #  39    0xb620c  1      OPC=nop             
  nop                                                       #  40    0xb620d  1      OPC=nop             
  nop                                                       #  41    0xb620e  1      OPC=nop             
  nop                                                       #  42    0xb620f  1      OPC=nop             
  nop                                                       #  43    0xb6210  1      OPC=nop             
  nop                                                       #  44    0xb6211  1      OPC=nop             
  nop                                                       #  45    0xb6212  1      OPC=nop             
  nop                                                       #  46    0xb6213  1      OPC=nop             
  nop                                                       #  47    0xb6214  1      OPC=nop             
  nop                                                       #  48    0xb6215  1      OPC=nop             
  nop                                                       #  49    0xb6216  1      OPC=nop             
  nop                                                       #  50    0xb6217  1      OPC=nop             
  nop                                                       #  51    0xb6218  1      OPC=nop             
  nop                                                       #  52    0xb6219  1      OPC=nop             
  nop                                                       #  53    0xb621a  1      OPC=nop             
  callq ._ZNSt11__timepunctIwE23_M_initialize_timepunctEPi  #  54    0xb621b  5      OPC=callq_label     
  addl $0x10, %esp                                          #  55    0xb6220  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                           #  56    0xb6223  3      OPC=addq_r64_r64    
  popq %rbx                                                 #  57    0xb6226  1      OPC=popq_r64_1      
  popq %r11                                                 #  58    0xb6227  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                   #  59    0xb6229  7      OPC=andl_r32_imm32  
  nop                                                       #  60    0xb6230  1      OPC=nop             
  nop                                                       #  61    0xb6231  1      OPC=nop             
  nop                                                       #  62    0xb6232  1      OPC=nop             
  nop                                                       #  63    0xb6233  1      OPC=nop             
  addq %r15, %r11                                           #  64    0xb6234  3      OPC=addq_r64_r64    
  jmpq %r11                                                 #  65    0xb6237  3      OPC=jmpq_r64        
  nop                                                       #  66    0xb623a  1      OPC=nop             
  nop                                                       #  67    0xb623b  1      OPC=nop             
  nop                                                       #  68    0xb623c  1      OPC=nop             
  nop                                                       #  69    0xb623d  1      OPC=nop             
  nop                                                       #  70    0xb623e  1      OPC=nop             
  nop                                                       #  71    0xb623f  1      OPC=nop             
  nop                                                       #  72    0xb6240  1      OPC=nop             
  nop                                                       #  73    0xb6241  1      OPC=nop             
  nop                                                       #  74    0xb6242  1      OPC=nop             
  nop                                                       #  75    0xb6243  1      OPC=nop             
  nop                                                       #  76    0xb6244  1      OPC=nop             
  nop                                                       #  77    0xb6245  1      OPC=nop             
  nop                                                       #  78    0xb6246  1      OPC=nop             
  movl %ebx, %edi                                           #  79    0xb6247  2      OPC=movl_r32_r32    
  movl %eax, 0x8(%rsp)                                      #  80    0xb6249  4      OPC=movl_m32_r32    
  nop                                                       #  81    0xb624d  1      OPC=nop             
  nop                                                       #  82    0xb624e  1      OPC=nop             
  nop                                                       #  83    0xb624f  1      OPC=nop             
  nop                                                       #  84    0xb6250  1      OPC=nop             
  nop                                                       #  85    0xb6251  1      OPC=nop             
  nop                                                       #  86    0xb6252  1      OPC=nop             
  nop                                                       #  87    0xb6253  1      OPC=nop             
  nop                                                       #  88    0xb6254  1      OPC=nop             
  nop                                                       #  89    0xb6255  1      OPC=nop             
  nop                                                       #  90    0xb6256  1      OPC=nop             
  nop                                                       #  91    0xb6257  1      OPC=nop             
  nop                                                       #  92    0xb6258  1      OPC=nop             
  nop                                                       #  93    0xb6259  1      OPC=nop             
  nop                                                       #  94    0xb625a  1      OPC=nop             
  nop                                                       #  95    0xb625b  1      OPC=nop             
  nop                                                       #  96    0xb625c  1      OPC=nop             
  nop                                                       #  97    0xb625d  1      OPC=nop             
  nop                                                       #  98    0xb625e  1      OPC=nop             
  nop                                                       #  99    0xb625f  1      OPC=nop             
  nop                                                       #  100   0xb6260  1      OPC=nop             
  nop                                                       #  101   0xb6261  1      OPC=nop             
  callq ._ZNSt6locale5facetD2Ev                             #  102   0xb6262  5      OPC=callq_label     
  movl 0x8(%rsp), %eax                                      #  103   0xb6267  4      OPC=movl_r32_m32    
  movl %eax, %edi                                           #  104   0xb626b  2      OPC=movl_r32_r32    
  nop                                                       #  105   0xb626d  1      OPC=nop             
  nop                                                       #  106   0xb626e  1      OPC=nop             
  nop                                                       #  107   0xb626f  1      OPC=nop             
  nop                                                       #  108   0xb6270  1      OPC=nop             
  nop                                                       #  109   0xb6271  1      OPC=nop             
  nop                                                       #  110   0xb6272  1      OPC=nop             
  nop                                                       #  111   0xb6273  1      OPC=nop             
  nop                                                       #  112   0xb6274  1      OPC=nop             
  nop                                                       #  113   0xb6275  1      OPC=nop             
  nop                                                       #  114   0xb6276  1      OPC=nop             
  nop                                                       #  115   0xb6277  1      OPC=nop             
  nop                                                       #  116   0xb6278  1      OPC=nop             
  nop                                                       #  117   0xb6279  1      OPC=nop             
  nop                                                       #  118   0xb627a  1      OPC=nop             
  nop                                                       #  119   0xb627b  1      OPC=nop             
  nop                                                       #  120   0xb627c  1      OPC=nop             
  nop                                                       #  121   0xb627d  1      OPC=nop             
  nop                                                       #  122   0xb627e  1      OPC=nop             
  nop                                                       #  123   0xb627f  1      OPC=nop             
  nop                                                       #  124   0xb6280  1      OPC=nop             
  nop                                                       #  125   0xb6281  1      OPC=nop             
  callq ._Unwind_Resume                                     #  126   0xb6282  5      OPC=callq_label     
                                                                                                         
.size _ZNSt11__timepunctIwEC1EPSt17__timepunct_cacheIwEj, .-_ZNSt11__timepunctIwEC1EPSt17__timepunct_cacheIwEj

