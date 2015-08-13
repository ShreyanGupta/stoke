  .text
  .globl _mbrtowc_r
  .type _mbrtowc_r, @function

#! file-offset 0x161c60
#! rip-offset  0x121c60
#! capacity    320 bytes

# Text                         #  Line  RIP       Bytes  Opcode              
._mbrtowc_r:                   #        0x121c60  0      OPC=<label>         
  movq %r12, -0x18(%rsp)       #  1     0x121c60  5      OPC=movq_m64_r64    
  movl %edi, %r12d             #  2     0x121c65  3      OPC=movl_r32_r32    
  movq %rbx, -0x20(%rsp)       #  3     0x121c68  5      OPC=movq_m64_r64    
  leal 0x104(%r12), %eax       #  4     0x121c6d  8      OPC=leal_r32_m16    
  movl %r8d, %ebx              #  5     0x121c75  3      OPC=movl_r32_r32    
  movq %r13, -0x10(%rsp)       #  6     0x121c78  5      OPC=movq_m64_r64    
  nop                          #  7     0x121c7d  1      OPC=nop             
  nop                          #  8     0x121c7e  1      OPC=nop             
  nop                          #  9     0x121c7f  1      OPC=nop             
  movq %r14, -0x8(%rsp)        #  10    0x121c80  5      OPC=movq_m64_r64    
  movl %edx, %edx              #  11    0x121c85  2      OPC=movl_r32_r32    
  subl $0x38, %esp             #  12    0x121c87  3      OPC=subl_r32_imm8   
  addq %r15, %rsp              #  13    0x121c8a  3      OPC=addq_r64_r64    
  testq %rbx, %rbx             #  14    0x121c8d  3      OPC=testq_r64_r64   
  movl %esi, %r13d             #  15    0x121c90  3      OPC=movl_r32_r32    
  cmoveq %rax, %rbx            #  16    0x121c93  4      OPC=cmoveq_r64_r64  
  testq %rdx, %rdx             #  17    0x121c97  3      OPC=testq_r64_r64   
  je .L_121d40                 #  18    0x121c9a  6      OPC=je_label_1      
  movl 0xff4f339(%rip), %r14d  #  19    0x121ca0  7      OPC=movl_r32_m32    
  movq %rdx, 0x8(%rsp)         #  20    0x121ca7  5      OPC=movq_m64_r64    
  movl %ecx, (%rsp)            #  21    0x121cac  3      OPC=movl_m32_r32    
  nop                          #  22    0x121caf  1      OPC=nop             
  nop                          #  23    0x121cb0  1      OPC=nop             
  nop                          #  24    0x121cb1  1      OPC=nop             
  nop                          #  25    0x121cb2  1      OPC=nop             
  nop                          #  26    0x121cb3  1      OPC=nop             
  nop                          #  27    0x121cb4  1      OPC=nop             
  nop                          #  28    0x121cb5  1      OPC=nop             
  nop                          #  29    0x121cb6  1      OPC=nop             
  nop                          #  30    0x121cb7  1      OPC=nop             
  nop                          #  31    0x121cb8  1      OPC=nop             
  nop                          #  32    0x121cb9  1      OPC=nop             
  nop                          #  33    0x121cba  1      OPC=nop             
  callq .__locale_charset      #  34    0x121cbb  5      OPC=callq_label     
  movl %ebx, %r9d              #  35    0x121cc0  3      OPC=movl_r32_r32    
  movl %eax, %r8d              #  36    0x121cc3  3      OPC=movl_r32_r32    
  movl (%rsp), %ecx            #  37    0x121cc6  3      OPC=movl_r32_m32    
  movq 0x8(%rsp), %rdx         #  38    0x121cc9  5      OPC=movq_r64_m64    
  movl %r13d, %esi             #  39    0x121cce  3      OPC=movl_r32_r32    
  movl %r12d, %edi             #  40    0x121cd1  3      OPC=movl_r32_r32    
  xchgw %ax, %ax               #  41    0x121cd4  2      OPC=xchgw_ax_r16    
  andl $0xffffffe0, %r14d      #  42    0x121cd6  7      OPC=andl_r32_imm32  
  nop                          #  43    0x121cdd  1      OPC=nop             
  nop                          #  44    0x121cde  1      OPC=nop             
  nop                          #  45    0x121cdf  1      OPC=nop             
  nop                          #  46    0x121ce0  1      OPC=nop             
  addq %r15, %r14              #  47    0x121ce1  3      OPC=addq_r64_r64    
  callq %r14                   #  48    0x121ce4  3      OPC=callq_r64       
.L_121ce0:                     #        0x121ce7  0      OPC=<label>         
  cmpl $0xffffffff, %eax       #  49    0x121ce7  6      OPC=cmpl_r32_imm32  
  nop                          #  50    0x121ced  1      OPC=nop             
  nop                          #  51    0x121cee  1      OPC=nop             
  nop                          #  52    0x121cef  1      OPC=nop             
  jne .L_121d00                #  53    0x121cf0  2      OPC=jne_label       
  movl %ebx, %ebx              #  54    0x121cf2  2      OPC=movl_r32_r32    
  movl $0x0, (%r15,%rbx,1)     #  55    0x121cf4  8      OPC=movl_m32_imm32  
  movl %r12d, %r12d            #  56    0x121cfc  3      OPC=movl_r32_r32    
  movl $0x54, (%r15,%r12,1)    #  57    0x121cff  8      OPC=movl_m32_imm32  
  nop                          #  58    0x121d07  1      OPC=nop             
  nop                          #  59    0x121d08  1      OPC=nop             
  nop                          #  60    0x121d09  1      OPC=nop             
  nop                          #  61    0x121d0a  1      OPC=nop             
  nop                          #  62    0x121d0b  1      OPC=nop             
  nop                          #  63    0x121d0c  1      OPC=nop             
.L_121d00:                     #        0x121d0d  0      OPC=<label>         
  movq 0x18(%rsp), %rbx        #  64    0x121d0d  5      OPC=movq_r64_m64    
  movq 0x20(%rsp), %r12        #  65    0x121d12  5      OPC=movq_r64_m64    
  movq 0x28(%rsp), %r13        #  66    0x121d17  5      OPC=movq_r64_m64    
  movq 0x30(%rsp), %r14        #  67    0x121d1c  5      OPC=movq_r64_m64    
  addl $0x38, %esp             #  68    0x121d21  3      OPC=addl_r32_imm8   
  addq %r15, %rsp              #  69    0x121d24  3      OPC=addq_r64_r64    
  popq %r11                    #  70    0x121d27  2      OPC=popq_r64_1      
  nop                          #  71    0x121d29  1      OPC=nop             
  nop                          #  72    0x121d2a  1      OPC=nop             
  nop                          #  73    0x121d2b  1      OPC=nop             
  nop                          #  74    0x121d2c  1      OPC=nop             
  andl $0xffffffe0, %r11d      #  75    0x121d2d  7      OPC=andl_r32_imm32  
  nop                          #  76    0x121d34  1      OPC=nop             
  nop                          #  77    0x121d35  1      OPC=nop             
  nop                          #  78    0x121d36  1      OPC=nop             
  nop                          #  79    0x121d37  1      OPC=nop             
  addq %r15, %r11              #  80    0x121d38  3      OPC=addq_r64_r64    
  jmpq %r11                    #  81    0x121d3b  3      OPC=jmpq_r64        
  nop                          #  82    0x121d3e  1      OPC=nop             
  nop                          #  83    0x121d3f  1      OPC=nop             
  nop                          #  84    0x121d40  1      OPC=nop             
  nop                          #  85    0x121d41  1      OPC=nop             
  nop                          #  86    0x121d42  1      OPC=nop             
  nop                          #  87    0x121d43  1      OPC=nop             
  nop                          #  88    0x121d44  1      OPC=nop             
  nop                          #  89    0x121d45  1      OPC=nop             
  nop                          #  90    0x121d46  1      OPC=nop             
  nop                          #  91    0x121d47  1      OPC=nop             
  nop                          #  92    0x121d48  1      OPC=nop             
  nop                          #  93    0x121d49  1      OPC=nop             
  nop                          #  94    0x121d4a  1      OPC=nop             
  nop                          #  95    0x121d4b  1      OPC=nop             
  nop                          #  96    0x121d4c  1      OPC=nop             
  nop                          #  97    0x121d4d  1      OPC=nop             
  nop                          #  98    0x121d4e  1      OPC=nop             
  nop                          #  99    0x121d4f  1      OPC=nop             
  nop                          #  100   0x121d50  1      OPC=nop             
  nop                          #  101   0x121d51  1      OPC=nop             
  nop                          #  102   0x121d52  1      OPC=nop             
  nop                          #  103   0x121d53  1      OPC=nop             
.L_121d40:                     #        0x121d54  0      OPC=<label>         
  movl 0xff4f299(%rip), %r13d  #  104   0x121d54  7      OPC=movl_r32_m32    
  nop                          #  105   0x121d5b  1      OPC=nop             
  nop                          #  106   0x121d5c  1      OPC=nop             
  nop                          #  107   0x121d5d  1      OPC=nop             
  nop                          #  108   0x121d5e  1      OPC=nop             
  nop                          #  109   0x121d5f  1      OPC=nop             
  nop                          #  110   0x121d60  1      OPC=nop             
  nop                          #  111   0x121d61  1      OPC=nop             
  nop                          #  112   0x121d62  1      OPC=nop             
  nop                          #  113   0x121d63  1      OPC=nop             
  nop                          #  114   0x121d64  1      OPC=nop             
  nop                          #  115   0x121d65  1      OPC=nop             
  nop                          #  116   0x121d66  1      OPC=nop             
  nop                          #  117   0x121d67  1      OPC=nop             
  nop                          #  118   0x121d68  1      OPC=nop             
  nop                          #  119   0x121d69  1      OPC=nop             
  nop                          #  120   0x121d6a  1      OPC=nop             
  nop                          #  121   0x121d6b  1      OPC=nop             
  nop                          #  122   0x121d6c  1      OPC=nop             
  nop                          #  123   0x121d6d  1      OPC=nop             
  nop                          #  124   0x121d6e  1      OPC=nop             
  callq .__locale_charset      #  125   0x121d6f  5      OPC=callq_label     
  movl %ebx, %r9d              #  126   0x121d74  3      OPC=movl_r32_r32    
  movl %eax, %r8d              #  127   0x121d77  3      OPC=movl_r32_r32    
  movl $0x1, %ecx              #  128   0x121d7a  5      OPC=movl_r32_imm32  
  movl $0x1003e92c, %edx       #  129   0x121d7f  5      OPC=movl_r32_imm32  
  xorl %esi, %esi              #  130   0x121d84  2      OPC=xorl_r32_r32    
  movl %r12d, %edi             #  131   0x121d86  3      OPC=movl_r32_r32    
  nop                          #  132   0x121d89  1      OPC=nop             
  andl $0xffffffe0, %r13d      #  133   0x121d8a  7      OPC=andl_r32_imm32  
  nop                          #  134   0x121d91  1      OPC=nop             
  nop                          #  135   0x121d92  1      OPC=nop             
  nop                          #  136   0x121d93  1      OPC=nop             
  nop                          #  137   0x121d94  1      OPC=nop             
  addq %r15, %r13              #  138   0x121d95  3      OPC=addq_r64_r64    
  callq %r13                   #  139   0x121d98  3      OPC=callq_r64       
  jmpq .L_121ce0               #  140   0x121d9b  5      OPC=jmpq_label_1    
  nop                          #  141   0x121da0  1      OPC=nop             
  nop                          #  142   0x121da1  1      OPC=nop             
  nop                          #  143   0x121da2  1      OPC=nop             
  nop                          #  144   0x121da3  1      OPC=nop             
  nop                          #  145   0x121da4  1      OPC=nop             
  nop                          #  146   0x121da5  1      OPC=nop             
  nop                          #  147   0x121da6  1      OPC=nop             
  nop                          #  148   0x121da7  1      OPC=nop             
  nop                          #  149   0x121da8  1      OPC=nop             
  nop                          #  150   0x121da9  1      OPC=nop             
  nop                          #  151   0x121daa  1      OPC=nop             
  nop                          #  152   0x121dab  1      OPC=nop             
  nop                          #  153   0x121dac  1      OPC=nop             
  nop                          #  154   0x121dad  1      OPC=nop             
  nop                          #  155   0x121dae  1      OPC=nop             
  nop                          #  156   0x121daf  1      OPC=nop             
  nop                          #  157   0x121db0  1      OPC=nop             
  nop                          #  158   0x121db1  1      OPC=nop             
  nop                          #  159   0x121db2  1      OPC=nop             
  nop                          #  160   0x121db3  1      OPC=nop             
  nop                          #  161   0x121db4  1      OPC=nop             
  nop                          #  162   0x121db5  1      OPC=nop             
  nop                          #  163   0x121db6  1      OPC=nop             
  nop                          #  164   0x121db7  1      OPC=nop             
  nop                          #  165   0x121db8  1      OPC=nop             
  nop                          #  166   0x121db9  1      OPC=nop             
  nop                          #  167   0x121dba  1      OPC=nop             
                                                                             
.size _mbrtowc_r, .-_mbrtowc_r

