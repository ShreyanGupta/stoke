  .text
  .globl size_of_encoded_value
  .type size_of_encoded_value, @function

#! file-offset 0x14ed60
#! rip-offset  0x10ed60
#! capacity    256 bytes

# Text                     #  Line  RIP       Bytes  Opcode              
.size_of_encoded_value:    #        0x10ed60  0      OPC=<label>         
  subl $0x8, %esp          #  1     0x10ed60  3      OPC=subl_r32_imm8   
  addq %r15, %rsp          #  2     0x10ed63  3      OPC=addq_r64_r64    
  cmpb $0xff, %dil         #  3     0x10ed66  4      OPC=cmpb_r8_imm8    
  je .L_10ee40             #  4     0x10ed6a  6      OPC=je_label_1      
  andl $0x7, %edi          #  5     0x10ed70  3      OPC=andl_r32_imm8   
  cmpl $0x2, %edi          #  6     0x10ed73  3      OPC=cmpl_r32_imm8   
  je .L_10ee20             #  7     0x10ed76  6      OPC=je_label_1      
  nop                      #  8     0x10ed7c  1      OPC=nop             
  nop                      #  9     0x10ed7d  1      OPC=nop             
  nop                      #  10    0x10ed7e  1      OPC=nop             
  nop                      #  11    0x10ed7f  1      OPC=nop             
  jle .L_10edc0            #  12    0x10ed80  2      OPC=jle_label       
  cmpl $0x3, %edi          #  13    0x10ed82  3      OPC=cmpl_r32_imm8   
  je .L_10ede0             #  14    0x10ed85  2      OPC=je_label        
  cmpl $0x4, %edi          #  15    0x10ed87  3      OPC=cmpl_r32_imm8   
  je .L_10ee00             #  16    0x10ed8a  2      OPC=je_label        
  nop                      #  17    0x10ed8c  1      OPC=nop             
  nop                      #  18    0x10ed8d  1      OPC=nop             
  nop                      #  19    0x10ed8e  1      OPC=nop             
  nop                      #  20    0x10ed8f  1      OPC=nop             
  nop                      #  21    0x10ed90  1      OPC=nop             
  nop                      #  22    0x10ed91  1      OPC=nop             
  nop                      #  23    0x10ed92  1      OPC=nop             
  nop                      #  24    0x10ed93  1      OPC=nop             
  nop                      #  25    0x10ed94  1      OPC=nop             
  nop                      #  26    0x10ed95  1      OPC=nop             
  nop                      #  27    0x10ed96  1      OPC=nop             
  nop                      #  28    0x10ed97  1      OPC=nop             
  nop                      #  29    0x10ed98  1      OPC=nop             
  nop                      #  30    0x10ed99  1      OPC=nop             
  nop                      #  31    0x10ed9a  1      OPC=nop             
  nop                      #  32    0x10ed9b  1      OPC=nop             
  nop                      #  33    0x10ed9c  1      OPC=nop             
  nop                      #  34    0x10ed9d  1      OPC=nop             
  nop                      #  35    0x10ed9e  1      OPC=nop             
  nop                      #  36    0x10ed9f  1      OPC=nop             
.L_10eda0:                 #        0x10eda0  0      OPC=<label>         
  nop                      #  37    0x10eda0  1      OPC=nop             
  nop                      #  38    0x10eda1  1      OPC=nop             
  nop                      #  39    0x10eda2  1      OPC=nop             
  nop                      #  40    0x10eda3  1      OPC=nop             
  nop                      #  41    0x10eda4  1      OPC=nop             
  nop                      #  42    0x10eda5  1      OPC=nop             
  nop                      #  43    0x10eda6  1      OPC=nop             
  nop                      #  44    0x10eda7  1      OPC=nop             
  nop                      #  45    0x10eda8  1      OPC=nop             
  nop                      #  46    0x10eda9  1      OPC=nop             
  nop                      #  47    0x10edaa  1      OPC=nop             
  nop                      #  48    0x10edab  1      OPC=nop             
  nop                      #  49    0x10edac  1      OPC=nop             
  nop                      #  50    0x10edad  1      OPC=nop             
  nop                      #  51    0x10edae  1      OPC=nop             
  nop                      #  52    0x10edaf  1      OPC=nop             
  nop                      #  53    0x10edb0  1      OPC=nop             
  nop                      #  54    0x10edb1  1      OPC=nop             
  nop                      #  55    0x10edb2  1      OPC=nop             
  nop                      #  56    0x10edb3  1      OPC=nop             
  nop                      #  57    0x10edb4  1      OPC=nop             
  nop                      #  58    0x10edb5  1      OPC=nop             
  nop                      #  59    0x10edb6  1      OPC=nop             
  nop                      #  60    0x10edb7  1      OPC=nop             
  nop                      #  61    0x10edb8  1      OPC=nop             
  nop                      #  62    0x10edb9  1      OPC=nop             
  nop                      #  63    0x10edba  1      OPC=nop             
  callq .abort             #  64    0x10edbb  5      OPC=callq_label     
.L_10edc0:                 #        0x10edc0  0      OPC=<label>         
  testl %edi, %edi         #  65    0x10edc0  2      OPC=testl_r32_r32   
  jne .L_10eda0            #  66    0x10edc2  2      OPC=jne_label       
  nop                      #  67    0x10edc4  1      OPC=nop             
  nop                      #  68    0x10edc5  1      OPC=nop             
  nop                      #  69    0x10edc6  1      OPC=nop             
  nop                      #  70    0x10edc7  1      OPC=nop             
  nop                      #  71    0x10edc8  1      OPC=nop             
  nop                      #  72    0x10edc9  1      OPC=nop             
  nop                      #  73    0x10edca  1      OPC=nop             
  nop                      #  74    0x10edcb  1      OPC=nop             
  nop                      #  75    0x10edcc  1      OPC=nop             
  nop                      #  76    0x10edcd  1      OPC=nop             
  nop                      #  77    0x10edce  1      OPC=nop             
  nop                      #  78    0x10edcf  1      OPC=nop             
  nop                      #  79    0x10edd0  1      OPC=nop             
  nop                      #  80    0x10edd1  1      OPC=nop             
  nop                      #  81    0x10edd2  1      OPC=nop             
  nop                      #  82    0x10edd3  1      OPC=nop             
  nop                      #  83    0x10edd4  1      OPC=nop             
  nop                      #  84    0x10edd5  1      OPC=nop             
  nop                      #  85    0x10edd6  1      OPC=nop             
  nop                      #  86    0x10edd7  1      OPC=nop             
  nop                      #  87    0x10edd8  1      OPC=nop             
  nop                      #  88    0x10edd9  1      OPC=nop             
  nop                      #  89    0x10edda  1      OPC=nop             
  nop                      #  90    0x10eddb  1      OPC=nop             
  nop                      #  91    0x10eddc  1      OPC=nop             
  nop                      #  92    0x10eddd  1      OPC=nop             
  nop                      #  93    0x10edde  1      OPC=nop             
  nop                      #  94    0x10eddf  1      OPC=nop             
.L_10ede0:                 #        0x10ede0  0      OPC=<label>         
  addl $0x8, %esp          #  95    0x10ede0  3      OPC=addl_r32_imm8   
  addq %r15, %rsp          #  96    0x10ede3  3      OPC=addq_r64_r64    
  movl $0x4, %eax          #  97    0x10ede6  5      OPC=movl_r32_imm32  
  popq %r11                #  98    0x10edeb  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d  #  99    0x10eded  7      OPC=andl_r32_imm32  
  nop                      #  100   0x10edf4  1      OPC=nop             
  nop                      #  101   0x10edf5  1      OPC=nop             
  nop                      #  102   0x10edf6  1      OPC=nop             
  nop                      #  103   0x10edf7  1      OPC=nop             
  addq %r15, %r11          #  104   0x10edf8  3      OPC=addq_r64_r64    
  jmpq %r11                #  105   0x10edfb  3      OPC=jmpq_r64        
  nop                      #  106   0x10edfe  1      OPC=nop             
  nop                      #  107   0x10edff  1      OPC=nop             
  nop                      #  108   0x10ee00  1      OPC=nop             
  nop                      #  109   0x10ee01  1      OPC=nop             
  nop                      #  110   0x10ee02  1      OPC=nop             
  nop                      #  111   0x10ee03  1      OPC=nop             
  nop                      #  112   0x10ee04  1      OPC=nop             
  nop                      #  113   0x10ee05  1      OPC=nop             
  nop                      #  114   0x10ee06  1      OPC=nop             
.L_10ee00:                 #        0x10ee07  0      OPC=<label>         
  addl $0x8, %esp          #  115   0x10ee07  3      OPC=addl_r32_imm8   
  addq %r15, %rsp          #  116   0x10ee0a  3      OPC=addq_r64_r64    
  movl $0x8, %eax          #  117   0x10ee0d  5      OPC=movl_r32_imm32  
  popq %r11                #  118   0x10ee12  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d  #  119   0x10ee14  7      OPC=andl_r32_imm32  
  nop                      #  120   0x10ee1b  1      OPC=nop             
  nop                      #  121   0x10ee1c  1      OPC=nop             
  nop                      #  122   0x10ee1d  1      OPC=nop             
  nop                      #  123   0x10ee1e  1      OPC=nop             
  addq %r15, %r11          #  124   0x10ee1f  3      OPC=addq_r64_r64    
  jmpq %r11                #  125   0x10ee22  3      OPC=jmpq_r64        
  nop                      #  126   0x10ee25  1      OPC=nop             
  nop                      #  127   0x10ee26  1      OPC=nop             
  nop                      #  128   0x10ee27  1      OPC=nop             
  nop                      #  129   0x10ee28  1      OPC=nop             
  nop                      #  130   0x10ee29  1      OPC=nop             
  nop                      #  131   0x10ee2a  1      OPC=nop             
  nop                      #  132   0x10ee2b  1      OPC=nop             
  nop                      #  133   0x10ee2c  1      OPC=nop             
  nop                      #  134   0x10ee2d  1      OPC=nop             
.L_10ee20:                 #        0x10ee2e  0      OPC=<label>         
  addl $0x8, %esp          #  135   0x10ee2e  3      OPC=addl_r32_imm8   
  addq %r15, %rsp          #  136   0x10ee31  3      OPC=addq_r64_r64    
  movl $0x2, %eax          #  137   0x10ee34  5      OPC=movl_r32_imm32  
  popq %r11                #  138   0x10ee39  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d  #  139   0x10ee3b  7      OPC=andl_r32_imm32  
  nop                      #  140   0x10ee42  1      OPC=nop             
  nop                      #  141   0x10ee43  1      OPC=nop             
  nop                      #  142   0x10ee44  1      OPC=nop             
  nop                      #  143   0x10ee45  1      OPC=nop             
  addq %r15, %r11          #  144   0x10ee46  3      OPC=addq_r64_r64    
  jmpq %r11                #  145   0x10ee49  3      OPC=jmpq_r64        
  nop                      #  146   0x10ee4c  1      OPC=nop             
  nop                      #  147   0x10ee4d  1      OPC=nop             
  nop                      #  148   0x10ee4e  1      OPC=nop             
  nop                      #  149   0x10ee4f  1      OPC=nop             
  nop                      #  150   0x10ee50  1      OPC=nop             
  nop                      #  151   0x10ee51  1      OPC=nop             
  nop                      #  152   0x10ee52  1      OPC=nop             
  nop                      #  153   0x10ee53  1      OPC=nop             
  nop                      #  154   0x10ee54  1      OPC=nop             
.L_10ee40:                 #        0x10ee55  0      OPC=<label>         
  addl $0x8, %esp          #  155   0x10ee55  3      OPC=addl_r32_imm8   
  addq %r15, %rsp          #  156   0x10ee58  3      OPC=addq_r64_r64    
  xorl %eax, %eax          #  157   0x10ee5b  2      OPC=xorl_r32_r32    
  popq %r11                #  158   0x10ee5d  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d  #  159   0x10ee5f  7      OPC=andl_r32_imm32  
  nop                      #  160   0x10ee66  1      OPC=nop             
  nop                      #  161   0x10ee67  1      OPC=nop             
  nop                      #  162   0x10ee68  1      OPC=nop             
  nop                      #  163   0x10ee69  1      OPC=nop             
  addq %r15, %r11          #  164   0x10ee6a  3      OPC=addq_r64_r64    
  jmpq %r11                #  165   0x10ee6d  3      OPC=jmpq_r64        
  nop                      #  166   0x10ee70  1      OPC=nop             
  nop                      #  167   0x10ee71  1      OPC=nop             
  nop                      #  168   0x10ee72  1      OPC=nop             
  nop                      #  169   0x10ee73  1      OPC=nop             
  nop                      #  170   0x10ee74  1      OPC=nop             
  nop                      #  171   0x10ee75  1      OPC=nop             
  nop                      #  172   0x10ee76  1      OPC=nop             
  nop                      #  173   0x10ee77  1      OPC=nop             
  nop                      #  174   0x10ee78  1      OPC=nop             
  nop                      #  175   0x10ee79  1      OPC=nop             
  nop                      #  176   0x10ee7a  1      OPC=nop             
  nop                      #  177   0x10ee7b  1      OPC=nop             
                                                                         
.size size_of_encoded_value, .-size_of_encoded_value

