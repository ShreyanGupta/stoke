  .text
  .globl __cxa_get_globals_fast
  .type __cxa_get_globals_fast, @function

#! file-offset 0x120940
#! rip-offset  0xe0940
#! capacity    192 bytes

# Text                          #  Line  RIP      Bytes  Opcode              
.__cxa_get_globals_fast:        #        0xe0940  0      OPC=<label>         
  subl $0x8, %esp               #  1     0xe0940  3      OPC=subl_r32_imm8   
  addq %r15, %rsp               #  2     0xe0943  3      OPC=addq_r64_r64    
  cmpb $0x0, 0xff92ceb(%rip)    #  3     0xe0946  7      OPC=cmpb_m8_imm8    
  movl $0x1007363c, %eax        #  4     0xe094d  5      OPC=movl_r32_imm32  
  jne .L_e0980                  #  5     0xe0952  2      OPC=jne_label       
  addl $0x8, %esp               #  6     0xe0954  3      OPC=addl_r32_imm8   
  addq %r15, %rsp               #  7     0xe0957  3      OPC=addq_r64_r64    
  popq %r11                     #  8     0xe095a  2      OPC=popq_r64_1      
  nop                           #  9     0xe095c  1      OPC=nop             
  nop                           #  10    0xe095d  1      OPC=nop             
  nop                           #  11    0xe095e  1      OPC=nop             
  nop                           #  12    0xe095f  1      OPC=nop             
  andl $0xffffffe0, %r11d       #  13    0xe0960  7      OPC=andl_r32_imm32  
  nop                           #  14    0xe0967  1      OPC=nop             
  nop                           #  15    0xe0968  1      OPC=nop             
  nop                           #  16    0xe0969  1      OPC=nop             
  nop                           #  17    0xe096a  1      OPC=nop             
  addq %r15, %r11               #  18    0xe096b  3      OPC=addq_r64_r64    
  jmpq %r11                     #  19    0xe096e  3      OPC=jmpq_r64        
  nop                           #  20    0xe0971  1      OPC=nop             
  nop                           #  21    0xe0972  1      OPC=nop             
  nop                           #  22    0xe0973  1      OPC=nop             
  nop                           #  23    0xe0974  1      OPC=nop             
  nop                           #  24    0xe0975  1      OPC=nop             
  nop                           #  25    0xe0976  1      OPC=nop             
  nop                           #  26    0xe0977  1      OPC=nop             
  nop                           #  27    0xe0978  1      OPC=nop             
  nop                           #  28    0xe0979  1      OPC=nop             
  nop                           #  29    0xe097a  1      OPC=nop             
  nop                           #  30    0xe097b  1      OPC=nop             
  nop                           #  31    0xe097c  1      OPC=nop             
  nop                           #  32    0xe097d  1      OPC=nop             
  nop                           #  33    0xe097e  1      OPC=nop             
  nop                           #  34    0xe097f  1      OPC=nop             
  nop                           #  35    0xe0980  1      OPC=nop             
  nop                           #  36    0xe0981  1      OPC=nop             
  nop                           #  37    0xe0982  1      OPC=nop             
  nop                           #  38    0xe0983  1      OPC=nop             
  nop                           #  39    0xe0984  1      OPC=nop             
  nop                           #  40    0xe0985  1      OPC=nop             
  nop                           #  41    0xe0986  1      OPC=nop             
.L_e0980:                       #        0xe0987  0      OPC=<label>         
  movl 0xff92cae(%rip), %edi    #  42    0xe0987  6      OPC=movl_r32_m32    
  nop                           #  43    0xe098d  1      OPC=nop             
  nop                           #  44    0xe098e  1      OPC=nop             
  nop                           #  45    0xe098f  1      OPC=nop             
  nop                           #  46    0xe0990  1      OPC=nop             
  nop                           #  47    0xe0991  1      OPC=nop             
  nop                           #  48    0xe0992  1      OPC=nop             
  nop                           #  49    0xe0993  1      OPC=nop             
  nop                           #  50    0xe0994  1      OPC=nop             
  nop                           #  51    0xe0995  1      OPC=nop             
  nop                           #  52    0xe0996  1      OPC=nop             
  nop                           #  53    0xe0997  1      OPC=nop             
  nop                           #  54    0xe0998  1      OPC=nop             
  nop                           #  55    0xe0999  1      OPC=nop             
  nop                           #  56    0xe099a  1      OPC=nop             
  nop                           #  57    0xe099b  1      OPC=nop             
  nop                           #  58    0xe099c  1      OPC=nop             
  nop                           #  59    0xe099d  1      OPC=nop             
  nop                           #  60    0xe099e  1      OPC=nop             
  nop                           #  61    0xe099f  1      OPC=nop             
  nop                           #  62    0xe09a0  1      OPC=nop             
  nop                           #  63    0xe09a1  1      OPC=nop             
  callq .pthread_getspecific    #  64    0xe09a2  5      OPC=callq_label     
  addl $0x8, %esp               #  65    0xe09a7  3      OPC=addl_r32_imm8   
  addq %r15, %rsp               #  66    0xe09aa  3      OPC=addq_r64_r64    
  movl %eax, %eax               #  67    0xe09ad  2      OPC=movl_r32_r32    
  popq %r11                     #  68    0xe09af  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d       #  69    0xe09b1  7      OPC=andl_r32_imm32  
  nop                           #  70    0xe09b8  1      OPC=nop             
  nop                           #  71    0xe09b9  1      OPC=nop             
  nop                           #  72    0xe09ba  1      OPC=nop             
  nop                           #  73    0xe09bb  1      OPC=nop             
  addq %r15, %r11               #  74    0xe09bc  3      OPC=addq_r64_r64    
  jmpq %r11                     #  75    0xe09bf  3      OPC=jmpq_r64        
  nop                           #  76    0xe09c2  1      OPC=nop             
  nop                           #  77    0xe09c3  1      OPC=nop             
  nop                           #  78    0xe09c4  1      OPC=nop             
  nop                           #  79    0xe09c5  1      OPC=nop             
  nop                           #  80    0xe09c6  1      OPC=nop             
  nop                           #  81    0xe09c7  1      OPC=nop             
  nop                           #  82    0xe09c8  1      OPC=nop             
  nop                           #  83    0xe09c9  1      OPC=nop             
  nop                           #  84    0xe09ca  1      OPC=nop             
  nop                           #  85    0xe09cb  1      OPC=nop             
  nop                           #  86    0xe09cc  1      OPC=nop             
  nop                           #  87    0xe09cd  1      OPC=nop             
  cmpq $0xff, %rdx              #  88    0xe09ce  4      OPC=cmpq_r64_imm8   
  movl %eax, %edi               #  89    0xe09d2  2      OPC=movl_r32_r32    
  je .L_e09e0                   #  90    0xe09d4  2      OPC=je_label        
  nop                           #  91    0xe09d6  1      OPC=nop             
  nop                           #  92    0xe09d7  1      OPC=nop             
  nop                           #  93    0xe09d8  1      OPC=nop             
  nop                           #  94    0xe09d9  1      OPC=nop             
  nop                           #  95    0xe09da  1      OPC=nop             
  nop                           #  96    0xe09db  1      OPC=nop             
  nop                           #  97    0xe09dc  1      OPC=nop             
  nop                           #  98    0xe09dd  1      OPC=nop             
  nop                           #  99    0xe09de  1      OPC=nop             
  nop                           #  100   0xe09df  1      OPC=nop             
  nop                           #  101   0xe09e0  1      OPC=nop             
  nop                           #  102   0xe09e1  1      OPC=nop             
  nop                           #  103   0xe09e2  1      OPC=nop             
  nop                           #  104   0xe09e3  1      OPC=nop             
  nop                           #  105   0xe09e4  1      OPC=nop             
  nop                           #  106   0xe09e5  1      OPC=nop             
  nop                           #  107   0xe09e6  1      OPC=nop             
  nop                           #  108   0xe09e7  1      OPC=nop             
  nop                           #  109   0xe09e8  1      OPC=nop             
  callq ._Unwind_Resume         #  110   0xe09e9  5      OPC=callq_label     
.L_e09e0:                       #        0xe09ee  0      OPC=<label>         
  nop                           #  111   0xe09ee  1      OPC=nop             
  nop                           #  112   0xe09ef  1      OPC=nop             
  nop                           #  113   0xe09f0  1      OPC=nop             
  nop                           #  114   0xe09f1  1      OPC=nop             
  nop                           #  115   0xe09f2  1      OPC=nop             
  nop                           #  116   0xe09f3  1      OPC=nop             
  nop                           #  117   0xe09f4  1      OPC=nop             
  nop                           #  118   0xe09f5  1      OPC=nop             
  nop                           #  119   0xe09f6  1      OPC=nop             
  nop                           #  120   0xe09f7  1      OPC=nop             
  nop                           #  121   0xe09f8  1      OPC=nop             
  nop                           #  122   0xe09f9  1      OPC=nop             
  nop                           #  123   0xe09fa  1      OPC=nop             
  nop                           #  124   0xe09fb  1      OPC=nop             
  nop                           #  125   0xe09fc  1      OPC=nop             
  nop                           #  126   0xe09fd  1      OPC=nop             
  nop                           #  127   0xe09fe  1      OPC=nop             
  nop                           #  128   0xe09ff  1      OPC=nop             
  nop                           #  129   0xe0a00  1      OPC=nop             
  nop                           #  130   0xe0a01  1      OPC=nop             
  nop                           #  131   0xe0a02  1      OPC=nop             
  nop                           #  132   0xe0a03  1      OPC=nop             
  nop                           #  133   0xe0a04  1      OPC=nop             
  nop                           #  134   0xe0a05  1      OPC=nop             
  nop                           #  135   0xe0a06  1      OPC=nop             
  nop                           #  136   0xe0a07  1      OPC=nop             
  nop                           #  137   0xe0a08  1      OPC=nop             
  callq .__cxa_call_unexpected  #  138   0xe0a09  5      OPC=callq_label     
                                                                             
.size __cxa_get_globals_fast, .-__cxa_get_globals_fast

