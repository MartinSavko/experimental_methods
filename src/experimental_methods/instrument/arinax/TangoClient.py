"""
  This code is provided AS IS for example purpose and testing MD Device Server
  ARINAX Sep. 2021
"""
import tango
from GenericClient import GenericClient, Attribute

class TangoClientFactory():
    @staticmethod
    def instantiate(*argc, **kwargs):
        return TangoClient(**kwargs)

class TangoClient(GenericClient):
    def __init__(self, address):
        GenericClient.__init__(self)
        self.server = tango.DeviceProxy(address)

    def hasCmd(self, cmdNameLower):
        return cmdNameLower.casefold() in (cmd.casefold() for cmd in self.server.get_command_list())

    def hasAttribute(self, attrNameLower):
        return attrNameLower.casefold() in (attr.casefold() for attr in self.server.get_attribute_list())

    def runCmd(self, cmdName, *args, **kwds):
        if len(args) >1:
            return self.server.command_inout(cmdName, cmd_param=[str(x) for x in args])
        else:
            return self.server.command_inout(cmdName, *args)

    def readAttribute(self, attrName):
        attrName = attrName.replace(" ", "")
        return self.server[attrName].value

    def writeAttribute(self, attrName, value):
        if not isinstance(value, str):
            try:
                self.server[attrName] = value[0]
            except:
                 self.server[attrName] = value
        self.server[attrName] = value

    def getAttributesList(self):
        ret = []
        # for attr in self.server.get_attribute_list():
        #     print(attr)
        # #     # print(self.server[attr])
        # #     # print(self.server.attribute_query(attr))
        #     data = self.server.read_attribute(attr)
        #     ret.append(
        #         Attribute(name=data.name, value=data.value, accessType=self.server.attribute_query(attr).writable is tango.AttrWriteType.READ_WRITE,
        #                   rType=data.data_format))
        # print("*************")
        for attr in self.server.attribute_list_query_ex():
            # print(attr.name)
            ret.append(Attribute(name = attr.name, value = 0, accessType = attr.writable is tango.AttrWriteType.READ_WRITE , rType = attr.data_format))
        return ret

    def onTangoEvent(self, event):
#        print(event.attr_name)
#        print(event.attr_value)
#        val=kw['value']
#        if "enum" in kw['type']:
#            val=kw['char_value']
        if not event.err:
            self.onEventReceived(event.attr_value.name, event.attr_value.value, event.attr_value.time)

    def nullWriter(txt):
        return

    def doSubscribe(self, attrName):
        return self.server.subscribe_event(attrName, tango.EventType.CHANGE_EVENT, self.onTangoEvent)

    def doUnsubscribe(self, attrName, eventId):
        self.server.unsubscribe_event(eventId)

    def hasEvents(self):
        return False
