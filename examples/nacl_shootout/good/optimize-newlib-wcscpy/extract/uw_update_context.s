  .text
  .globl uw_update_context
  .type uw_update_context, @function

#! file-offset 0x14ca80
#! rip-offset  0x10ca80
#! capacity    320 bytes

# Text                            #  Line  RIP       Bytes  Opcode              
.uw_update_context:               #        0x10ca80  0      OPC=<label>         
  pushq %r12                      #  1     0x10ca80  2      OPC=pushq_r64_1     
  movl %esi, %r12d                #  2     0x10ca82  3      OPC=movl_r32_r32    
  movl %r12d, %esi                #  3     0x10ca85  3      OPC=movl_r32_r32    
  pushq %rbx                      #  4     0x10ca88  1      OPC=pushq_r64_1     
  movl %edi, %ebx                 #  5     0x10ca89  2      OPC=movl_r32_r32    
  movl %ebx, %edi                 #  6     0x10ca8b  2      OPC=movl_r32_r32    
  subl $0x8, %esp                 #  7     0x10ca8d  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                 #  8     0x10ca90  3      OPC=addq_r64_r64    
  nop                             #  9     0x10ca93  1      OPC=nop             
  nop                             #  10    0x10ca94  1      OPC=nop             
  nop                             #  11    0x10ca95  1      OPC=nop             
  nop                             #  12    0x10ca96  1      OPC=nop             
  nop                             #  13    0x10ca97  1      OPC=nop             
  nop                             #  14    0x10ca98  1      OPC=nop             
  nop                             #  15    0x10ca99  1      OPC=nop             
  nop                             #  16    0x10ca9a  1      OPC=nop             
  callq .uw_update_context_1      #  17    0x10ca9b  5      OPC=callq_label     
  movl %r12d, %r12d               #  18    0x10caa0  3      OPC=movl_r32_r32    
  movl 0x158(%r15,%r12,1), %eax   #  19    0x10caa3  8      OPC=movl_r32_m32    
  movl %eax, %edx                 #  20    0x10caab  2      OPC=movl_r32_r32    
  shll $0x4, %edx                 #  21    0x10caad  3      OPC=shll_r32_imm8   
  movslq %edx, %rdx               #  22    0x10cab0  3      OPC=movslq_r64_r32  
  addq %rdx, %r12                 #  23    0x10cab3  3      OPC=addq_r64_r64    
  movl %r12d, %r12d               #  24    0x10cab6  3      OPC=movl_r32_r32    
  cmpl $0x6, 0x8(%r15,%r12,1)     #  25    0x10cab9  6      OPC=cmpl_m32_imm8   
  nop                             #  26    0x10cabf  1      OPC=nop             
  je .L_10cb60                    #  27    0x10cac0  6      OPC=je_label_1      
  cmpl $0x11, %eax                #  28    0x10cac6  3      OPC=cmpl_r32_imm8   
  jg .L_10cba0                    #  29    0x10cac9  6      OPC=jg_label_1      
  leal 0xff6c02b(%rip), %ecx      #  30    0x10cacf  6      OPC=leal_r32_m16    
  movslq %eax, %rdx               #  31    0x10cad5  3      OPC=movslq_r64_r32  
  movl %ebx, %ebx                 #  32    0x10cad8  2      OPC=movl_r32_r32    
  testb $0x40, 0x67(%r15,%rbx,1)  #  33    0x10cada  6      OPC=testb_m8_imm8   
  leaq (%rcx,%rdx,1), %rdx        #  34    0x10cae0  4      OPC=leaq_r64_m16    
  movl %edx, %edx                 #  35    0x10cae4  2      OPC=movl_r32_r32    
  movzbl (%r15,%rdx,1), %esi      #  36    0x10cae6  5      OPC=movzbl_r32_m8   
  leal (,%rax,4), %edx            #  37    0x10caeb  7      OPC=leal_r32_m16    
  movslq %edx, %rdx               #  38    0x10caf2  3      OPC=movslq_r64_r32  
  leaq (%rbx,%rdx,1), %rdx        #  39    0x10caf5  4      OPC=leaq_r64_m16    
  movl %edx, %edx                 #  40    0x10caf9  2      OPC=movl_r32_r32    
  movl (%r15,%rdx,1), %ecx        #  41    0x10cafb  4      OPC=movl_r32_m32    
  nop                             #  42    0x10caff  1      OPC=nop             
  je .L_10cb20                    #  43    0x10cb00  2      OPC=je_label        
  addl $0x70, %eax                #  44    0x10cb02  3      OPC=addl_r32_imm8   
  movq %rcx, %rdx                 #  45    0x10cb05  3      OPC=movq_r64_r64    
  cltq                            #  46    0x10cb08  2      OPC=cltq            
  leaq (%rbx,%rax,1), %rax        #  47    0x10cb0a  4      OPC=leaq_r64_m16    
  movl %eax, %eax                 #  48    0x10cb0e  2      OPC=movl_r32_r32    
  cmpb $0x0, 0x8(%r15,%rax,1)     #  49    0x10cb10  6      OPC=cmpb_m8_imm8    
  jne .L_10cb40                   #  50    0x10cb16  2      OPC=jne_label       
  nop                             #  51    0x10cb18  1      OPC=nop             
  nop                             #  52    0x10cb19  1      OPC=nop             
  nop                             #  53    0x10cb1a  1      OPC=nop             
  nop                             #  54    0x10cb1b  1      OPC=nop             
  nop                             #  55    0x10cb1c  1      OPC=nop             
  nop                             #  56    0x10cb1d  1      OPC=nop             
  nop                             #  57    0x10cb1e  1      OPC=nop             
  nop                             #  58    0x10cb1f  1      OPC=nop             
.L_10cb20:                        #        0x10cb20  0      OPC=<label>         
  movzbl %sil, %eax               #  59    0x10cb20  4      OPC=movzbl_r32_r8   
  cmpl $0x4, %eax                 #  60    0x10cb24  3      OPC=cmpl_r32_imm8   
  je .L_10cb80                    #  61    0x10cb27  2      OPC=je_label        
  cmpl $0x8, %eax                 #  62    0x10cb29  3      OPC=cmpl_r32_imm8   
  jne .L_10cba0                   #  63    0x10cb2c  2      OPC=jne_label       
  movl %ecx, %ecx                 #  64    0x10cb2e  2      OPC=movl_r32_r32    
  movq (%r15,%rcx,1), %rdx        #  65    0x10cb30  4      OPC=movq_r64_m64    
  nop                             #  66    0x10cb34  1      OPC=nop             
  nop                             #  67    0x10cb35  1      OPC=nop             
  nop                             #  68    0x10cb36  1      OPC=nop             
  nop                             #  69    0x10cb37  1      OPC=nop             
  nop                             #  70    0x10cb38  1      OPC=nop             
  nop                             #  71    0x10cb39  1      OPC=nop             
  nop                             #  72    0x10cb3a  1      OPC=nop             
  nop                             #  73    0x10cb3b  1      OPC=nop             
  nop                             #  74    0x10cb3c  1      OPC=nop             
  nop                             #  75    0x10cb3d  1      OPC=nop             
  nop                             #  76    0x10cb3e  1      OPC=nop             
  nop                             #  77    0x10cb3f  1      OPC=nop             
.L_10cb40:                        #        0x10cb40  0      OPC=<label>         
  movl %ebx, %ebx                 #  78    0x10cb40  2      OPC=movl_r32_r32    
  movl %edx, 0x4c(%r15,%rbx,1)    #  79    0x10cb42  5      OPC=movl_m32_r32    
  addl $0x8, %esp                 #  80    0x10cb47  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                 #  81    0x10cb4a  3      OPC=addq_r64_r64    
  popq %rbx                       #  82    0x10cb4d  1      OPC=popq_r64_1      
  popq %r12                       #  83    0x10cb4e  2      OPC=popq_r64_1      
  popq %r11                       #  84    0x10cb50  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d         #  85    0x10cb52  7      OPC=andl_r32_imm32  
  nop                             #  86    0x10cb59  1      OPC=nop             
  nop                             #  87    0x10cb5a  1      OPC=nop             
  nop                             #  88    0x10cb5b  1      OPC=nop             
  nop                             #  89    0x10cb5c  1      OPC=nop             
  addq %r15, %r11                 #  90    0x10cb5d  3      OPC=addq_r64_r64    
  jmpq %r11                       #  91    0x10cb60  3      OPC=jmpq_r64        
  nop                             #  92    0x10cb63  1      OPC=nop             
  nop                             #  93    0x10cb64  1      OPC=nop             
  nop                             #  94    0x10cb65  1      OPC=nop             
  nop                             #  95    0x10cb66  1      OPC=nop             
.L_10cb60:                        #        0x10cb67  0      OPC=<label>         
  movl %ebx, %ebx                 #  96    0x10cb67  2      OPC=movl_r32_r32    
  movl $0x0, 0x4c(%r15,%rbx,1)    #  97    0x10cb69  9      OPC=movl_m32_imm32  
  addl $0x8, %esp                 #  98    0x10cb72  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                 #  99    0x10cb75  3      OPC=addq_r64_r64    
  popq %rbx                       #  100   0x10cb78  1      OPC=popq_r64_1      
  popq %r12                       #  101   0x10cb79  2      OPC=popq_r64_1      
  popq %r11                       #  102   0x10cb7b  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d         #  103   0x10cb7d  7      OPC=andl_r32_imm32  
  nop                             #  104   0x10cb84  1      OPC=nop             
  nop                             #  105   0x10cb85  1      OPC=nop             
  nop                             #  106   0x10cb86  1      OPC=nop             
  nop                             #  107   0x10cb87  1      OPC=nop             
  addq %r15, %r11                 #  108   0x10cb88  3      OPC=addq_r64_r64    
  jmpq %r11                       #  109   0x10cb8b  3      OPC=jmpq_r64        
.L_10cb80:                        #        0x10cb8e  0      OPC=<label>         
  movl %ecx, %ecx                 #  110   0x10cb8e  2      OPC=movl_r32_r32    
  movl (%r15,%rcx,1), %edx        #  111   0x10cb90  4      OPC=movl_r32_m32    
  jmpq .L_10cb40                  #  112   0x10cb94  2      OPC=jmpq_label      
  nop                             #  113   0x10cb96  1      OPC=nop             
  nop                             #  114   0x10cb97  1      OPC=nop             
  nop                             #  115   0x10cb98  1      OPC=nop             
  nop                             #  116   0x10cb99  1      OPC=nop             
  nop                             #  117   0x10cb9a  1      OPC=nop             
  nop                             #  118   0x10cb9b  1      OPC=nop             
  nop                             #  119   0x10cb9c  1      OPC=nop             
  nop                             #  120   0x10cb9d  1      OPC=nop             
  nop                             #  121   0x10cb9e  1      OPC=nop             
  nop                             #  122   0x10cb9f  1      OPC=nop             
  nop                             #  123   0x10cba0  1      OPC=nop             
  nop                             #  124   0x10cba1  1      OPC=nop             
  nop                             #  125   0x10cba2  1      OPC=nop             
  nop                             #  126   0x10cba3  1      OPC=nop             
  nop                             #  127   0x10cba4  1      OPC=nop             
  nop                             #  128   0x10cba5  1      OPC=nop             
  nop                             #  129   0x10cba6  1      OPC=nop             
  nop                             #  130   0x10cba7  1      OPC=nop             
  nop                             #  131   0x10cba8  1      OPC=nop             
  nop                             #  132   0x10cba9  1      OPC=nop             
  nop                             #  133   0x10cbaa  1      OPC=nop             
  nop                             #  134   0x10cbab  1      OPC=nop             
  nop                             #  135   0x10cbac  1      OPC=nop             
  nop                             #  136   0x10cbad  1      OPC=nop             
.L_10cba0:                        #        0x10cbae  0      OPC=<label>         
  nop                             #  137   0x10cbae  1      OPC=nop             
  nop                             #  138   0x10cbaf  1      OPC=nop             
  nop                             #  139   0x10cbb0  1      OPC=nop             
  nop                             #  140   0x10cbb1  1      OPC=nop             
  nop                             #  141   0x10cbb2  1      OPC=nop             
  nop                             #  142   0x10cbb3  1      OPC=nop             
  nop                             #  143   0x10cbb4  1      OPC=nop             
  nop                             #  144   0x10cbb5  1      OPC=nop             
  nop                             #  145   0x10cbb6  1      OPC=nop             
  nop                             #  146   0x10cbb7  1      OPC=nop             
  nop                             #  147   0x10cbb8  1      OPC=nop             
  nop                             #  148   0x10cbb9  1      OPC=nop             
  nop                             #  149   0x10cbba  1      OPC=nop             
  nop                             #  150   0x10cbbb  1      OPC=nop             
  nop                             #  151   0x10cbbc  1      OPC=nop             
  nop                             #  152   0x10cbbd  1      OPC=nop             
  nop                             #  153   0x10cbbe  1      OPC=nop             
  nop                             #  154   0x10cbbf  1      OPC=nop             
  nop                             #  155   0x10cbc0  1      OPC=nop             
  nop                             #  156   0x10cbc1  1      OPC=nop             
  nop                             #  157   0x10cbc2  1      OPC=nop             
  nop                             #  158   0x10cbc3  1      OPC=nop             
  nop                             #  159   0x10cbc4  1      OPC=nop             
  nop                             #  160   0x10cbc5  1      OPC=nop             
  nop                             #  161   0x10cbc6  1      OPC=nop             
  nop                             #  162   0x10cbc7  1      OPC=nop             
  nop                             #  163   0x10cbc8  1      OPC=nop             
  callq .abort                    #  164   0x10cbc9  5      OPC=callq_label     
                                                                                
.size uw_update_context, .-uw_update_context

