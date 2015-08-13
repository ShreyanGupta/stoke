  .text
  .globl run_wmemcmp
  .type run_wmemcmp, @function

#! file-offset 0x69e20
#! rip-offset  0x29e20
#! capacity    288 bytes

# Text                       #  Line  RIP      Bytes  Opcode              
.run_wmemcmp:                #        0x29e20  0      OPC=<label>         
  pushq %r14                 #  1     0x29e20  2      OPC=pushq_r64_1     
  pushq %r13                 #  2     0x29e22  2      OPC=pushq_r64_1     
  pushq %r12                 #  3     0x29e24  2      OPC=pushq_r64_1     
  pushq %rbx                 #  4     0x29e26  1      OPC=pushq_r64_1     
  subl $0x2018, %esp         #  5     0x29e27  6      OPC=subl_r32_imm32  
  addq %r15, %rsp            #  6     0x29e2d  3      OPC=addq_r64_r64    
  movl %edi, %r14d           #  7     0x29e30  3      OPC=movl_r32_r32    
  leal 0x1010(%rsp), %r13d   #  8     0x29e33  8      OPC=leal_r32_m16    
  movl %r13d, %eax           #  9     0x29e3b  3      OPC=movl_r32_r32    
  xchgw %ax, %ax             #  10    0x29e3e  2      OPC=xchgw_ax_r16    
  leal 0x10(%rsp), %edx      #  11    0x29e40  4      OPC=leal_r32_m16    
  movq %rdx, 0x8(%rsp)       #  12    0x29e44  5      OPC=movq_m64_r64    
  movl 0x8(%rsp), %edx       #  13    0x29e49  4      OPC=movl_r32_m32    
  leal 0x1000(%r13), %ebx    #  14    0x29e4d  7      OPC=leal_r32_m16    
  nop                        #  15    0x29e54  1      OPC=nop             
  nop                        #  16    0x29e55  1      OPC=nop             
  nop                        #  17    0x29e56  1      OPC=nop             
  nop                        #  18    0x29e57  1      OPC=nop             
  nop                        #  19    0x29e58  1      OPC=nop             
  nop                        #  20    0x29e59  1      OPC=nop             
  nop                        #  21    0x29e5a  1      OPC=nop             
  nop                        #  22    0x29e5b  1      OPC=nop             
  nop                        #  23    0x29e5c  1      OPC=nop             
  nop                        #  24    0x29e5d  1      OPC=nop             
  nop                        #  25    0x29e5e  1      OPC=nop             
  nop                        #  26    0x29e5f  1      OPC=nop             
.L_29e60:                    #        0x29e60  0      OPC=<label>         
  movl %eax, %ecx            #  27    0x29e60  2      OPC=movl_r32_r32    
  movl %ecx, %ecx            #  28    0x29e62  2      OPC=movl_r32_r32    
  movl $0x61, (%r15,%rcx,1)  #  29    0x29e64  8      OPC=movl_m32_imm32  
  movl %edx, %ecx            #  30    0x29e6c  2      OPC=movl_r32_r32    
  movl %ecx, %ecx            #  31    0x29e6e  2      OPC=movl_r32_r32    
  movl $0x61, (%r15,%rcx,1)  #  32    0x29e70  8      OPC=movl_m32_imm32  
  addl $0x4, %eax            #  33    0x29e78  3      OPC=addl_r32_imm8   
  addl $0x4, %edx            #  34    0x29e7b  3      OPC=addl_r32_imm8   
  cmpl %ebx, %eax            #  35    0x29e7e  2      OPC=cmpl_r32_r32    
  jne .L_29e60               #  36    0x29e80  2      OPC=jne_label       
  testl %r14d, %r14d         #  37    0x29e82  3      OPC=testl_r32_r32   
  je .L_29f20                #  38    0x29e85  6      OPC=je_label_1      
  movl $0x0, 0x200c(%rsp)    #  39    0x29e8b  11     OPC=movl_m32_imm32  
  nop                        #  40    0x29e96  1      OPC=nop             
  nop                        #  41    0x29e97  1      OPC=nop             
  nop                        #  42    0x29e98  1      OPC=nop             
  nop                        #  43    0x29e99  1      OPC=nop             
  nop                        #  44    0x29e9a  1      OPC=nop             
  nop                        #  45    0x29e9b  1      OPC=nop             
  nop                        #  46    0x29e9c  1      OPC=nop             
  nop                        #  47    0x29e9d  1      OPC=nop             
  nop                        #  48    0x29e9e  1      OPC=nop             
  nop                        #  49    0x29e9f  1      OPC=nop             
  movl $0x0, 0x100c(%rsp)    #  50    0x29ea0  11     OPC=movl_m32_imm32  
  xorl %ebx, %ebx            #  51    0x29eab  2      OPC=xorl_r32_r32    
  xorl %r12d, %r12d          #  52    0x29ead  3      OPC=xorl_r32_r32    
  nop                        #  53    0x29eb0  1      OPC=nop             
  nop                        #  54    0x29eb1  1      OPC=nop             
  nop                        #  55    0x29eb2  1      OPC=nop             
  nop                        #  56    0x29eb3  1      OPC=nop             
  nop                        #  57    0x29eb4  1      OPC=nop             
  nop                        #  58    0x29eb5  1      OPC=nop             
  nop                        #  59    0x29eb6  1      OPC=nop             
  nop                        #  60    0x29eb7  1      OPC=nop             
  nop                        #  61    0x29eb8  1      OPC=nop             
  nop                        #  62    0x29eb9  1      OPC=nop             
  nop                        #  63    0x29eba  1      OPC=nop             
  nop                        #  64    0x29ebb  1      OPC=nop             
  nop                        #  65    0x29ebc  1      OPC=nop             
  nop                        #  66    0x29ebd  1      OPC=nop             
  nop                        #  67    0x29ebe  1      OPC=nop             
  nop                        #  68    0x29ebf  1      OPC=nop             
.L_29ec0:                    #        0x29ec0  0      OPC=<label>         
  movl $0x400, %edx          #  69    0x29ec0  5      OPC=movl_r32_imm32  
  movl 0x8(%rsp), %esi       #  70    0x29ec5  4      OPC=movl_r32_m32    
  movl %r13d, %edi           #  71    0x29ec9  3      OPC=movl_r32_r32    
  xorl %eax, %eax            #  72    0x29ecc  2      OPC=xorl_r32_r32    
  nop                        #  73    0x29ece  1      OPC=nop             
  nop                        #  74    0x29ecf  1      OPC=nop             
  nop                        #  75    0x29ed0  1      OPC=nop             
  nop                        #  76    0x29ed1  1      OPC=nop             
  nop                        #  77    0x29ed2  1      OPC=nop             
  nop                        #  78    0x29ed3  1      OPC=nop             
  nop                        #  79    0x29ed4  1      OPC=nop             
  nop                        #  80    0x29ed5  1      OPC=nop             
  nop                        #  81    0x29ed6  1      OPC=nop             
  nop                        #  82    0x29ed7  1      OPC=nop             
  nop                        #  83    0x29ed8  1      OPC=nop             
  nop                        #  84    0x29ed9  1      OPC=nop             
  nop                        #  85    0x29eda  1      OPC=nop             
  callq .wmemcmp             #  86    0x29edb  5      OPC=callq_label     
  addl %eax, %r12d           #  87    0x29ee0  3      OPC=addl_r32_r32    
  addl $0x1, %ebx            #  88    0x29ee3  3      OPC=addl_r32_imm8   
  cmpl %r14d, %ebx           #  89    0x29ee6  3      OPC=cmpl_r32_r32    
  jb .L_29ec0                #  90    0x29ee9  2      OPC=jb_label        
  nop                        #  91    0x29eeb  1      OPC=nop             
  nop                        #  92    0x29eec  1      OPC=nop             
  nop                        #  93    0x29eed  1      OPC=nop             
  nop                        #  94    0x29eee  1      OPC=nop             
  nop                        #  95    0x29eef  1      OPC=nop             
  nop                        #  96    0x29ef0  1      OPC=nop             
  nop                        #  97    0x29ef1  1      OPC=nop             
  nop                        #  98    0x29ef2  1      OPC=nop             
  nop                        #  99    0x29ef3  1      OPC=nop             
  nop                        #  100   0x29ef4  1      OPC=nop             
  nop                        #  101   0x29ef5  1      OPC=nop             
  nop                        #  102   0x29ef6  1      OPC=nop             
  nop                        #  103   0x29ef7  1      OPC=nop             
  nop                        #  104   0x29ef8  1      OPC=nop             
  nop                        #  105   0x29ef9  1      OPC=nop             
  nop                        #  106   0x29efa  1      OPC=nop             
  nop                        #  107   0x29efb  1      OPC=nop             
  nop                        #  108   0x29efc  1      OPC=nop             
  nop                        #  109   0x29efd  1      OPC=nop             
  nop                        #  110   0x29efe  1      OPC=nop             
  nop                        #  111   0x29eff  1      OPC=nop             
.L_29f00:                    #        0x29f00  0      OPC=<label>         
  movl %r12d, %eax           #  112   0x29f00  3      OPC=movl_r32_r32    
  addl $0x2018, %esp         #  113   0x29f03  6      OPC=addl_r32_imm32  
  addq %r15, %rsp            #  114   0x29f09  3      OPC=addq_r64_r64    
  popq %rbx                  #  115   0x29f0c  1      OPC=popq_r64_1      
  popq %r12                  #  116   0x29f0d  2      OPC=popq_r64_1      
  popq %r13                  #  117   0x29f0f  2      OPC=popq_r64_1      
  popq %r14                  #  118   0x29f11  2      OPC=popq_r64_1      
  popq %r11                  #  119   0x29f13  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d    #  120   0x29f15  7      OPC=andl_r32_imm32  
  nop                        #  121   0x29f1c  1      OPC=nop             
  nop                        #  122   0x29f1d  1      OPC=nop             
  nop                        #  123   0x29f1e  1      OPC=nop             
  nop                        #  124   0x29f1f  1      OPC=nop             
  addq %r15, %r11            #  125   0x29f20  3      OPC=addq_r64_r64    
  jmpq %r11                  #  126   0x29f23  3      OPC=jmpq_r64        
  nop                        #  127   0x29f26  1      OPC=nop             
.L_29f20:                    #        0x29f27  0      OPC=<label>         
  xorl %r12d, %r12d          #  128   0x29f27  3      OPC=xorl_r32_r32    
  jmpq .L_29f00              #  129   0x29f2a  2      OPC=jmpq_label      
  nop                        #  130   0x29f2c  1      OPC=nop             
  nop                        #  131   0x29f2d  1      OPC=nop             
  nop                        #  132   0x29f2e  1      OPC=nop             
  nop                        #  133   0x29f2f  1      OPC=nop             
  nop                        #  134   0x29f30  1      OPC=nop             
  nop                        #  135   0x29f31  1      OPC=nop             
  nop                        #  136   0x29f32  1      OPC=nop             
  nop                        #  137   0x29f33  1      OPC=nop             
  nop                        #  138   0x29f34  1      OPC=nop             
  nop                        #  139   0x29f35  1      OPC=nop             
  nop                        #  140   0x29f36  1      OPC=nop             
  nop                        #  141   0x29f37  1      OPC=nop             
  nop                        #  142   0x29f38  1      OPC=nop             
  nop                        #  143   0x29f39  1      OPC=nop             
  nop                        #  144   0x29f3a  1      OPC=nop             
  nop                        #  145   0x29f3b  1      OPC=nop             
  nop                        #  146   0x29f3c  1      OPC=nop             
  nop                        #  147   0x29f3d  1      OPC=nop             
  nop                        #  148   0x29f3e  1      OPC=nop             
  nop                        #  149   0x29f3f  1      OPC=nop             
  nop                        #  150   0x29f40  1      OPC=nop             
  nop                        #  151   0x29f41  1      OPC=nop             
  nop                        #  152   0x29f42  1      OPC=nop             
  nop                        #  153   0x29f43  1      OPC=nop             
  nop                        #  154   0x29f44  1      OPC=nop             
  nop                        #  155   0x29f45  1      OPC=nop             
  nop                        #  156   0x29f46  1      OPC=nop             
                                                                          
.size run_wmemcmp, .-run_wmemcmp

