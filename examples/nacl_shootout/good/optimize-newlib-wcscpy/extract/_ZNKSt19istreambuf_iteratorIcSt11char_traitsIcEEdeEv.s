  .text
  .globl _ZNKSt19istreambuf_iteratorIcSt11char_traitsIcEEdeEv
  .type _ZNKSt19istreambuf_iteratorIcSt11char_traitsIcEEdeEv, @function

#! file-offset 0xc01e0
#! rip-offset  0x801e0
#! capacity    224 bytes

# Text                                                  #  Line  RIP      Bytes  Opcode                
._ZNKSt19istreambuf_iteratorIcSt11char_traitsIcEEdeEv:  #        0x801e0  0      OPC=<label>           
  pushq %rbx                                            #  1     0x801e0  1      OPC=pushq_r64_1       
  movl %edi, %ebx                                       #  2     0x801e1  2      OPC=movl_r32_r32      
  movl $0xffffffff, %eax                                #  3     0x801e3  6      OPC=movl_r32_imm32_1  
  movl %ebx, %ebx                                       #  4     0x801e9  2      OPC=movl_r32_r32      
  movl (%r15,%rbx,1), %edi                              #  5     0x801eb  4      OPC=movl_r32_m32      
  testq %rdi, %rdi                                      #  6     0x801ef  3      OPC=testq_r64_r64     
  je .L_80220                                           #  7     0x801f2  2      OPC=je_label          
  movl %ebx, %ebx                                       #  8     0x801f4  2      OPC=movl_r32_r32      
  movl 0x4(%r15,%rbx,1), %edx                           #  9     0x801f6  5      OPC=movl_r32_m32      
  cmpl $0xffffffff, %edx                                #  10    0x801fb  6      OPC=cmpl_r32_imm32    
  nop                                                   #  11    0x80201  1      OPC=nop               
  nop                                                   #  12    0x80202  1      OPC=nop               
  nop                                                   #  13    0x80203  1      OPC=nop               
  movl %edx, %eax                                       #  14    0x80204  2      OPC=movl_r32_r32      
  nop                                                   #  15    0x80206  1      OPC=nop               
  je .L_80240                                           #  16    0x80207  2      OPC=je_label          
  nop                                                   #  17    0x80209  1      OPC=nop               
  nop                                                   #  18    0x8020a  1      OPC=nop               
  nop                                                   #  19    0x8020b  1      OPC=nop               
  nop                                                   #  20    0x8020c  1      OPC=nop               
  nop                                                   #  21    0x8020d  1      OPC=nop               
  nop                                                   #  22    0x8020e  1      OPC=nop               
  nop                                                   #  23    0x8020f  1      OPC=nop               
  nop                                                   #  24    0x80210  1      OPC=nop               
  nop                                                   #  25    0x80211  1      OPC=nop               
  nop                                                   #  26    0x80212  1      OPC=nop               
  nop                                                   #  27    0x80213  1      OPC=nop               
  nop                                                   #  28    0x80214  1      OPC=nop               
  nop                                                   #  29    0x80215  1      OPC=nop               
  nop                                                   #  30    0x80216  1      OPC=nop               
  nop                                                   #  31    0x80217  1      OPC=nop               
  nop                                                   #  32    0x80218  1      OPC=nop               
  nop                                                   #  33    0x80219  1      OPC=nop               
  nop                                                   #  34    0x8021a  1      OPC=nop               
  nop                                                   #  35    0x8021b  1      OPC=nop               
  nop                                                   #  36    0x8021c  1      OPC=nop               
  nop                                                   #  37    0x8021d  1      OPC=nop               
  nop                                                   #  38    0x8021e  1      OPC=nop               
  nop                                                   #  39    0x8021f  1      OPC=nop               
  nop                                                   #  40    0x80220  1      OPC=nop               
  nop                                                   #  41    0x80221  1      OPC=nop               
  nop                                                   #  42    0x80222  1      OPC=nop               
  nop                                                   #  43    0x80223  1      OPC=nop               
  nop                                                   #  44    0x80224  1      OPC=nop               
  nop                                                   #  45    0x80225  1      OPC=nop               
  nop                                                   #  46    0x80226  1      OPC=nop               
.L_80220:                                               #        0x80227  0      OPC=<label>           
  popq %rbx                                             #  47    0x80227  1      OPC=popq_r64_1        
  popq %r11                                             #  48    0x80228  2      OPC=popq_r64_1        
  andl $0xffffffe0, %r11d                               #  49    0x8022a  7      OPC=andl_r32_imm32    
  nop                                                   #  50    0x80231  1      OPC=nop               
  nop                                                   #  51    0x80232  1      OPC=nop               
  nop                                                   #  52    0x80233  1      OPC=nop               
  nop                                                   #  53    0x80234  1      OPC=nop               
  addq %r15, %r11                                       #  54    0x80235  3      OPC=addq_r64_r64      
  jmpq %r11                                             #  55    0x80238  3      OPC=jmpq_r64          
  nop                                                   #  56    0x8023b  1      OPC=nop               
  nop                                                   #  57    0x8023c  1      OPC=nop               
  nop                                                   #  58    0x8023d  1      OPC=nop               
  nop                                                   #  59    0x8023e  1      OPC=nop               
  nop                                                   #  60    0x8023f  1      OPC=nop               
  nop                                                   #  61    0x80240  1      OPC=nop               
  nop                                                   #  62    0x80241  1      OPC=nop               
  nop                                                   #  63    0x80242  1      OPC=nop               
  nop                                                   #  64    0x80243  1      OPC=nop               
  nop                                                   #  65    0x80244  1      OPC=nop               
  nop                                                   #  66    0x80245  1      OPC=nop               
  nop                                                   #  67    0x80246  1      OPC=nop               
  nop                                                   #  68    0x80247  1      OPC=nop               
  nop                                                   #  69    0x80248  1      OPC=nop               
  nop                                                   #  70    0x80249  1      OPC=nop               
  nop                                                   #  71    0x8024a  1      OPC=nop               
  nop                                                   #  72    0x8024b  1      OPC=nop               
  nop                                                   #  73    0x8024c  1      OPC=nop               
  nop                                                   #  74    0x8024d  1      OPC=nop               
.L_80240:                                               #        0x8024e  0      OPC=<label>           
  movl %edi, %edi                                       #  75    0x8024e  2      OPC=movl_r32_r32      
  movl 0x8(%r15,%rdi,1), %eax                           #  76    0x80250  5      OPC=movl_r32_m32      
  movl %edi, %edi                                       #  77    0x80255  2      OPC=movl_r32_r32      
  cmpl %eax, 0xc(%r15,%rdi,1)                           #  78    0x80257  5      OPC=cmpl_m32_r32      
  jbe .L_80280                                          #  79    0x8025c  2      OPC=jbe_label         
  movl %eax, %eax                                       #  80    0x8025e  2      OPC=movl_r32_r32      
  movzbl (%r15,%rax,1), %eax                            #  81    0x80260  5      OPC=movzbl_r32_m8     
  nop                                                   #  82    0x80265  1      OPC=nop               
  nop                                                   #  83    0x80266  1      OPC=nop               
  nop                                                   #  84    0x80267  1      OPC=nop               
  nop                                                   #  85    0x80268  1      OPC=nop               
  nop                                                   #  86    0x80269  1      OPC=nop               
  nop                                                   #  87    0x8026a  1      OPC=nop               
  nop                                                   #  88    0x8026b  1      OPC=nop               
  nop                                                   #  89    0x8026c  1      OPC=nop               
  nop                                                   #  90    0x8026d  1      OPC=nop               
.L_80260:                                               #        0x8026e  0      OPC=<label>           
  movl %ebx, %ebx                                       #  91    0x8026e  2      OPC=movl_r32_r32      
  movl %eax, 0x4(%r15,%rbx,1)                           #  92    0x80270  5      OPC=movl_m32_r32      
  popq %rbx                                             #  93    0x80275  1      OPC=popq_r64_1        
  popq %r11                                             #  94    0x80276  2      OPC=popq_r64_1        
  andl $0xffffffe0, %r11d                               #  95    0x80278  7      OPC=andl_r32_imm32    
  nop                                                   #  96    0x8027f  1      OPC=nop               
  nop                                                   #  97    0x80280  1      OPC=nop               
  nop                                                   #  98    0x80281  1      OPC=nop               
  nop                                                   #  99    0x80282  1      OPC=nop               
  addq %r15, %r11                                       #  100   0x80283  3      OPC=addq_r64_r64      
  jmpq %r11                                             #  101   0x80286  3      OPC=jmpq_r64          
  nop                                                   #  102   0x80289  1      OPC=nop               
  nop                                                   #  103   0x8028a  1      OPC=nop               
  nop                                                   #  104   0x8028b  1      OPC=nop               
  nop                                                   #  105   0x8028c  1      OPC=nop               
  nop                                                   #  106   0x8028d  1      OPC=nop               
  nop                                                   #  107   0x8028e  1      OPC=nop               
  nop                                                   #  108   0x8028f  1      OPC=nop               
  nop                                                   #  109   0x80290  1      OPC=nop               
  nop                                                   #  110   0x80291  1      OPC=nop               
  nop                                                   #  111   0x80292  1      OPC=nop               
  nop                                                   #  112   0x80293  1      OPC=nop               
  nop                                                   #  113   0x80294  1      OPC=nop               
.L_80280:                                               #        0x80295  0      OPC=<label>           
  movl %edi, %edi                                       #  114   0x80295  2      OPC=movl_r32_r32      
  movl (%r15,%rdi,1), %eax                              #  115   0x80297  4      OPC=movl_r32_m32      
  movl %eax, %eax                                       #  116   0x8029b  2      OPC=movl_r32_r32      
  movl 0x24(%r15,%rax,1), %eax                          #  117   0x8029d  5      OPC=movl_r32_m32      
  nop                                                   #  118   0x802a2  1      OPC=nop               
  nop                                                   #  119   0x802a3  1      OPC=nop               
  nop                                                   #  120   0x802a4  1      OPC=nop               
  nop                                                   #  121   0x802a5  1      OPC=nop               
  nop                                                   #  122   0x802a6  1      OPC=nop               
  nop                                                   #  123   0x802a7  1      OPC=nop               
  nop                                                   #  124   0x802a8  1      OPC=nop               
  nop                                                   #  125   0x802a9  1      OPC=nop               
  nop                                                   #  126   0x802aa  1      OPC=nop               
  nop                                                   #  127   0x802ab  1      OPC=nop               
  nop                                                   #  128   0x802ac  1      OPC=nop               
  andl $0xffffffe0, %eax                                #  129   0x802ad  6      OPC=andl_r32_imm32    
  nop                                                   #  130   0x802b3  1      OPC=nop               
  nop                                                   #  131   0x802b4  1      OPC=nop               
  nop                                                   #  132   0x802b5  1      OPC=nop               
  addq %r15, %rax                                       #  133   0x802b6  3      OPC=addq_r64_r64      
  callq %rax                                            #  134   0x802b9  2      OPC=callq_r64         
  cmpl $0xffffffff, %eax                                #  135   0x802bb  6      OPC=cmpl_r32_imm32    
  nop                                                   #  136   0x802c1  1      OPC=nop               
  nop                                                   #  137   0x802c2  1      OPC=nop               
  nop                                                   #  138   0x802c3  1      OPC=nop               
  jne .L_80260                                          #  139   0x802c4  2      OPC=jne_label         
  movl %ebx, %ebx                                       #  140   0x802c6  2      OPC=movl_r32_r32      
  movl $0x0, (%r15,%rbx,1)                              #  141   0x802c8  8      OPC=movl_m32_imm32    
  jmpq .L_80220                                         #  142   0x802d0  5      OPC=jmpq_label_1      
  nop                                                   #  143   0x802d5  1      OPC=nop               
  nop                                                   #  144   0x802d6  1      OPC=nop               
  nop                                                   #  145   0x802d7  1      OPC=nop               
  nop                                                   #  146   0x802d8  1      OPC=nop               
  nop                                                   #  147   0x802d9  1      OPC=nop               
  nop                                                   #  148   0x802da  1      OPC=nop               
  nop                                                   #  149   0x802db  1      OPC=nop               
  nop                                                   #  150   0x802dc  1      OPC=nop               
  nop                                                   #  151   0x802dd  1      OPC=nop               
  nop                                                   #  152   0x802de  1      OPC=nop               
  nop                                                   #  153   0x802df  1      OPC=nop               
  nop                                                   #  154   0x802e0  1      OPC=nop               
                                                                                                       
.size _ZNKSt19istreambuf_iteratorIcSt11char_traitsIcEEdeEv, .-_ZNKSt19istreambuf_iteratorIcSt11char_traitsIcEEdeEv

